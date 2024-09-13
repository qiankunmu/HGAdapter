import math

import torch
import torch.nn as nn
from adapters.modeling import Adapter, ParallelAdapter

from .HyperStructAdapterConfig import HyperStructAdapterConfig


class HyperStructAdapter(Adapter):
    # v2
    def __init__(self, adapter_name, input_size, down_sample, config: HyperStructAdapterConfig, *args, **kwargs):
        super().__init__(adapter_name, input_size, down_sample, config)

        self.use_hyper = config["use_hyper"]
        self.num_edge_types = config["num_edge_types"]
        self.use_norm = config["use_norm"]

        self.num_heads = config.num_heads
        self.head_size = self.down_sample // self.num_heads
        assert self.down_sample == self.head_size * self.num_heads, f"down sample {self.down_sample} can not be divisible by num heads {self.num_heads}"
        self.dropout_rate = config["dropout"]
        self.torch_dtype = getattr(torch, config["torch_dtype"])

        if self.use_hyper:
            self.Q1 = MultiAttnVector(self.num_edge_types, self.num_heads, self.head_size, self.torch_dtype)
            self.K1 = nn.Linear(self.down_sample, self.down_sample, dtype=self.torch_dtype)
            self.V1 = nn.Linear(self.down_sample, self.down_sample, dtype=self.torch_dtype)

            self.trans = HeteroLinear(self.down_sample, self.down_sample, self.num_edge_types,
                                      torch_dtype=self.torch_dtype)

            self.Q2 = nn.Linear(self.down_sample, self.down_sample, dtype=self.torch_dtype)
            self.K2 = nn.Linear(self.down_sample, self.down_sample, dtype=self.torch_dtype)
            self.V2 = nn.Linear(self.down_sample, self.down_sample, dtype=self.torch_dtype)

            self.dropout = nn.Dropout(p=self.dropout_rate)

        if self.use_norm:
            self.layer_norm = nn.LayerNorm(self.down_sample)

    def forward(self, x, residual_input, output_gating=False, hyperedge_indexs=None, edge_types=None, **kwargs):
        batch_size = x.size(0)

        if self.use_hyper and hyperedge_indexs is not None:
            # new add
            if self.add_layer_norm_before:
                x = self.adapter_norm_before(x)

            x = self.adapter_down[-2](x)
            # [batch_size, seq_len+1, down_sample]

            x0 = x
            x = self.dropout(x)

            hyperedge_indexs_s = hyperedge_indexs[:, 0, :]
            hyperedge_indexs_t = hyperedge_indexs[:, 1, :]
            # hyperedge_indexs_st [batch_size, num_relation]
            key1 = self.K1(x)
            key1 = torch.gather(key1, 1, hyperedge_indexs_s.unsqueeze(-1).expand(-1, -1, self.down_sample))
            # [batch_size, num_relation, down_sample]
            key1 = key1.reshape(batch_size, -1, self.num_heads, self.head_size)
            # [batch_size, num_relation, num_heads, head_size]
            value1 = self.V1(x)
            value1 = torch.gather(value1, 1, hyperedge_indexs_s.unsqueeze(-1).expand(-1, -1, self.down_sample))
            value1 = value1.reshape(batch_size, -1, self.num_heads, self.head_size)

            attn_score = self.Q1(key1, edge_types, hyperedge_indexs_t)
            # [batch_size, num_relation, num_heads]
            attn_score = attn_score.unsqueeze(-1)
            # [batch_size, num_relation, num_heads, 1]

            agg = value1 * attn_score
            # [batch_size, num_relation, num_heads, head_size]
            agg = agg.reshape(batch_size, -1, self.down_sample)
            # [batch_size, num_relation, down_sample]
            hyperedges = torch.zeros(batch_size, int(hyperedge_indexs_t.max()) + 1, self.down_sample,
                                     dtype=self.torch_dtype, device=x.device). \
                scatter_add_(1, hyperedge_indexs_t.unsqueeze(-1).expand(-1, -1, self.down_sample), agg)
            # [batch_size, num_edges+1, down_sample]

            hyperedges = torch.gather(hyperedges, 1, hyperedge_indexs_t.unsqueeze(-1).expand(-1, -1, self.down_sample))
            # [batch_size, num_relation, down_sample]
            hyperedges = self.trans(hyperedges, edge_types)

            key2 = self.K2(hyperedges)
            key2 = key2.reshape(batch_size, -1, self.num_heads, self.head_size)

            value2 = self.V2(hyperedges)
            value2 = value2.reshape(batch_size, -1, self.num_heads, self.head_size)

            query2 = self.Q2(x)
            query2 = torch.gather(query2, 1, hyperedge_indexs_s.unsqueeze(-1).expand(-1, -1, self.down_sample))
            query2 = query2.reshape(batch_size, -1, self.num_heads, self.head_size)
            # [batch_size, num_relation, num_heads, head_size]

            attn2 = (query2 * key2).sum(dim=-1)
            attn2 = attn2 / math.sqrt(self.head_size)
            # [batch_size, num_relation, num_heads]
            attn_score2 = softmax(attn2, hyperedge_indexs_s, x.size(1), self.torch_dtype)
            attn_score2 = attn_score2.unsqueeze(-1)
            # [batch_size, num_relation, num_heads, 1]

            agg2 = value2 * attn_score2
            agg2 = agg2.reshape(batch_size, -1, self.down_sample)
            # [batch_size, num_relation, down_sample]

            down = torch.zeros(batch_size, x.size(1), self.down_sample, dtype=self.torch_dtype,
                               device=x.device).scatter_add_ \
                (1, hyperedge_indexs_s.unsqueeze(-1).expand(-1, -1, self.down_sample), agg2)
            # [batch_size, seq_len+1, down_sample]

            if self.use_norm:
                down = self.layer_norm(down)

            # down = self.dropout(down)
            down = down + x0

            down = self.non_linearity(down)
            # new add end
        else:
            down = self.adapter_down(x)

        up = self.adapter_up(down)
        up = up * self.scaling
        output = up

        if self.use_gating:
            # x.shape = (batch_size, seq_len, hidden_size)
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            output = output * gate

        # apply residual connection before layer norm if configured in this way
        if self.adapter_residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        # if residual should be applied after layer norm, apply it here
        if not self.adapter_residual_before_ln:
            output = output + residual_input

        if self.use_gating and output_gating:
            return output, down, up, gate
        return output, down, up


class ParallelHyperStructAdapter(HyperStructAdapter, ParallelAdapter):
    def __init__(self, adapter_name, input_size, down_sample, config: HyperStructAdapterConfig):
        HyperStructAdapter.__init__(self, adapter_name, input_size, down_sample, config)

    def pre_forward(self, hidden_states, input_tensor, layer_norm, fusion_config=None):
        return ParallelAdapter.pre_forward(self, hidden_states, input_tensor, layer_norm, fusion_config)

    def forward(self, x, residual_input, output_gating=False, hyperedge_indexs=None, edge_types=None, **kwargs):
        batch_size = x.size(0)

        if self.use_hyper:
            # new add
            if self.add_layer_norm_before:
                x = self.adapter_norm_before(x)

            x = self.adapter_down[-2](x)
            # [batch_size, seq_len+1, down_sample]

            hyperedge_indexs_s = hyperedge_indexs[:, 0, :]
            hyperedge_indexs_t = hyperedge_indexs[:, 1, :]
            # hyperedge_indexs_st [batch_size, num_relation]
            key1 = self.K1(x)
            key1 = torch.gather(key1, 1, hyperedge_indexs_s.unsqueeze(-1).expand(-1, -1, self.down_sample))
            # [batch_size, num_relation, down_sample]
            key1 = key1.reshape(batch_size, -1, self.num_heads, self.head_size)
            # [batch_size, num_relation, num_heads, head_size]
            value1 = self.V1(x)
            value1 = torch.gather(value1, 1, hyperedge_indexs_s.unsqueeze(-1).expand(-1, -1, self.down_sample))
            value1 = value1.reshape(batch_size, -1, self.num_heads, self.head_size)

            attn_score = self.Q1(key1, edge_types, hyperedge_indexs_t)
            # [batch_size, num_relation, num_heads]
            attn_score = attn_score.unsqueeze(-1)
            # [batch_size, num_relation, num_heads, 1]

            agg = value1 * attn_score
            # [batch_size, num_relation, num_heads, head_size]
            agg = agg.reshape(batch_size, -1, self.down_sample)
            # [batch_size, num_relation, down_sample]
            hyperedges = torch.zeros(batch_size, int(hyperedge_indexs_t.max()) + 1, self.down_sample,
                                     dtype=self.torch_dtype, device=x.device). \
                scatter_add_(1, hyperedge_indexs_t.unsqueeze(-1).expand(-1, -1, self.down_sample), agg)
            # [batch_size, num_edges+1, down_sample]

            hyperedges = torch.gather(hyperedges, 1, hyperedge_indexs_t.unsqueeze(-1).expand(-1, -1, self.down_sample))
            # [batch_size, num_relation, down_sample]
            hyperedges = self.trans(hyperedges, edge_types)

            key2 = self.K2(hyperedges)
            key2 = key2.reshape(batch_size, -1, self.num_heads, self.head_size)

            value2 = self.V2(hyperedges)
            value2 = value2.reshape(batch_size, -1, self.num_heads, self.head_size)

            query2 = self.Q2(x)
            query2 = torch.gather(query2, 1, hyperedge_indexs_s.unsqueeze(-1).expand(-1, -1, self.down_sample))
            query2 = query2.reshape(batch_size, -1, self.num_heads, self.head_size)
            # [batch_size, num_relation, num_heads, head_size]

            attn2 = (query2 * key2).sum(dim=-1)
            attn2 = attn2 / math.sqrt(self.head_size)
            # [batch_size, num_relation, num_heads]
            attn_score2 = softmax(attn2, hyperedge_indexs_s, x.size(1), self.torch_dtype)
            attn_score2 = attn_score2.unsqueeze(-1)
            # [batch_size, num_relation, num_heads, 1]

            agg2 = value2 * attn_score2
            agg2 = agg2.reshape(batch_size, -1, self.down_sample)
            # [batch_size, num_relation, down_sample]

            down = torch.zeros(batch_size, x.size(1), self.down_sample, dtype=self.torch_dtype,
                               device=x.device).scatter_add_ \
                (1, hyperedge_indexs_s.unsqueeze(-1).expand(-1, -1, self.down_sample), agg2)
            # [batch_size, seq_len+1, down_sample]
            down = down + x

            down = self.non_linearity(down)
            # new add end
        else:
            down = self.adapter_down(x)

        up = self.adapter_up(down)
        up = up * self.scaling
        output = up

        if self.use_gating:
            # x.shape = (batch_size, seq_len, hidden_size)
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            output = output * gate

        # apply residual connection before layer norm if configured in this way
        if self.adapter_residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        # if residual should be applied after layer norm, apply it here
        if not self.adapter_residual_before_ln:
            output = output + residual_input

        if self.use_gating and output_gating:
            return output, down, up, gate
        return output, down, up

    def post_forward(self, hidden_states, input_hidden_states, input_tensor, layer_norm):
        return ParallelAdapter.post_forward(self, hidden_states, input_hidden_states, input_tensor, layer_norm)


class MultiAttnVector(nn.Module):
    def __init__(self, num_types, num_heads, head_size, torch_dtype=torch.float):
        super(MultiAttnVector, self).__init__()
        self.num_types = num_types
        self.num_heads = num_heads
        self.head_size = head_size
        self.torch_dtype = torch_dtype

        # self.attn_vector = [nn.Parameter(torch.Tensor(1, self.num_heads, self.head_size)) for i in range(self.num_types)]
        self.attn_vector = nn.Parameter(
            torch.Tensor(self.num_types, 1, self.num_heads, self.head_size).type(torch_dtype))

        self.reset_parameters()

    def reset_parameters(self):
        # for attn_vector in self.attn_vector:
        #     nn.init.xavier_uniform_(attn_vector)
        nn.init.xavier_uniform_(self.attn_vector)

    def forward(self, x, types, indexs):
        # x [batch_size, num_relation, num_heads, head_size] types, indexs [batch_size, num_relation]
        attns = x.new_empty(x.size(0), x.size(1), self.num_heads, dtype=self.torch_dtype)
        for i in range(self.num_types):
            attn_vector = self.attn_vector[i, :, :, :]
            mask = types == i
            # print("i=", i)
            # print("types=", types.size())
            # print("mask=", mask.size())
            # print("x=",x.size())
            t = x[mask]
            # [num, num_heads, head_size]

            attn = t * attn_vector
            attn = attn.sum(-1)
            # [num, num_heads]
            attns[mask] = attn

        attns = attns / math.sqrt(x.size(-1))
        # attns [batch_size, num_relation, num_heads]
        attn_score = softmax(attns, indexs, torch_dtype=self.torch_dtype)

        return attn_score


# refer to torch_geometric.utils.softmax
def softmax(src: torch.Tensor, indexs: torch.Tensor, num_t=None, torch_dtype=torch.float):
    if not num_t:
        num_t = int(indexs.max()) + 1

    indexs = indexs.unsqueeze(-1).expand(-1, -1, src.size(2))
    # src, indexs [batch_size, num_relation, num_heads]
    src_max = torch.zeros(src.size(0), num_t, src.size(2), dtype=torch_dtype, device=src.device).scatter_reduce_ \
        (1, indexs, src, reduce="amax", include_self=False)
    # src_max [batch_size, num_edges+1, num_heads]
    src_max = src_max.gather(1, indexs)
    # src_max [batch_size, num_relation, num_heads]
    out = (src - src_max).exp()
    # [batch_size, num_relation, num_heads]
    out_sum = torch.zeros(src.size(0), num_t, src.size(2), dtype=torch_dtype, device=src.device).scatter_add_(1, indexs,
                                                                                                              out)
    # [batch_size, num_edges+1, num_heads]
    out_sum = out_sum.gather(1, indexs)
    # [batch_size, num_relation, num_heads]

    return out / (out_sum + 1e-16)


# refer to torch_geometric.nn.HeteroLinear
class HeteroLinear(nn.Module):
    def __init__(self, in_features, out_features, num_types, torch_dtype=torch.float, **kwargs):
        super(HeteroLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_types = num_types
        self.torch_dtype = torch_dtype

        self.linear = nn.ModuleList(
            [nn.Linear(in_features, out_features, dtype=torch_dtype, **kwargs) for i in range(num_types)])

    def forward(self, x, types):
        # x [batch_size, num_relation, down_sample] types [batch_size, num_relation]
        out = x.new_empty(x.size(0), x.size(1), self.out_features, dtype=self.torch_dtype)
        for i, lin in enumerate(self.linear):
            mask = types == i
            out[mask] = lin(x[mask])
        # out [batch_size, num_relation, down_sample]
        return out


class HGbdAdapter(Adapter):
    # v2
    def __init__(self, adapter_name, input_size, down_sample, config: HyperStructAdapterConfig, *args, **kwargs):
        super().__init__(adapter_name, input_size, down_sample, config)

        self.use_hyper = config["use_hyper"]
        self.num_edge_types = config["num_edge_types"]
        self.use_norm = config["use_norm"]

        self.num_heads = config.num_heads
        self.head_size = self.down_sample // self.num_heads
        assert self.down_sample == self.head_size * self.num_heads, f"down sample {self.down_sample} can not be divisible by num heads {self.num_heads}"
        self.dropout_rate = config["dropout"]
        self.torch_dtype = getattr(torch, config["torch_dtype"])
        self.use_adapter2 = config["use_adapter2"]

        if self.use_hyper or self.use_adapter2:
            self.feedforward_down = nn.Linear(self.input_size, self.down_sample, dtype=self.torch_dtype)
            self.feedforward_up = nn.Linear(self.down_sample, self.input_size, dtype=self.torch_dtype)

            if self.use_hyper:
                self.Q1 = MultiAttnVector(self.num_edge_types, self.num_heads, self.head_size, self.torch_dtype)
                self.K1 = nn.Linear(self.down_sample, self.down_sample, dtype=self.torch_dtype)
                self.V1 = nn.Linear(self.down_sample, self.down_sample, dtype=self.torch_dtype)

                self.trans = HeteroLinear(self.down_sample, self.down_sample, self.num_edge_types,
                                          torch_dtype=self.torch_dtype)

                self.Q2 = nn.Linear(self.down_sample, self.down_sample, dtype=self.torch_dtype)
                self.K2 = nn.Linear(self.down_sample, self.down_sample, dtype=self.torch_dtype)
                self.V2 = nn.Linear(self.down_sample, self.down_sample, dtype=self.torch_dtype)

                self.dropout = nn.Dropout(p=self.dropout_rate)

        if self.use_norm:
            self.layer_norm = nn.LayerNorm(self.down_sample)

    def forward(self, x, residual_input, output_gating=False, hyperedge_indexs=None, edge_types=None, source_mask=None,
                **kwargs):
        batch_size = x.size(0)

        down = self.adapter_down(x)
        up1 = self.adapter_up(down)
        if source_mask is not None:
            up1 = up1.masked_fill_(source_mask, 0)

        if self.use_hyper and hyperedge_indexs is not None:
            # new add
            if self.add_layer_norm_before:
                x = self.adapter_norm_before(x)

            x = self.feedforward_down(x)
            # [batch_size, seq_len+1, down_sample]

            hyperedge_indexs_s = hyperedge_indexs[:, 0, :]
            hyperedge_indexs_t = hyperedge_indexs[:, 1, :]
            # hyperedge_indexs_st [batch_size, num_relation]
            key1 = self.K1(x)
            key1 = torch.gather(key1, 1, hyperedge_indexs_s.unsqueeze(-1).expand(-1, -1, self.down_sample))
            # [batch_size, num_relation, down_sample]
            key1 = key1.reshape(batch_size, -1, self.num_heads, self.head_size)
            # [batch_size, num_relation, num_heads, head_size]
            value1 = self.V1(x)
            value1 = torch.gather(value1, 1, hyperedge_indexs_s.unsqueeze(-1).expand(-1, -1, self.down_sample))
            value1 = value1.reshape(batch_size, -1, self.num_heads, self.head_size)

            attn_score = self.Q1(key1, edge_types, hyperedge_indexs_t)
            # [batch_size, num_relation, num_heads]
            attn_score = attn_score.unsqueeze(-1)
            # [batch_size, num_relation, num_heads, 1]

            agg = value1 * attn_score
            # [batch_size, num_relation, num_heads, head_size]
            agg = agg.reshape(batch_size, -1, self.down_sample)
            # [batch_size, num_relation, down_sample]
            hyperedges = torch.zeros(batch_size, int(hyperedge_indexs_t.max()) + 1, self.down_sample,
                                     dtype=self.torch_dtype, device=x.device). \
                scatter_add_(1, hyperedge_indexs_t.unsqueeze(-1).expand(-1, -1, self.down_sample), agg)
            # [batch_size, num_edges+1, down_sample]

            hyperedges = torch.gather(hyperedges, 1, hyperedge_indexs_t.unsqueeze(-1).expand(-1, -1, self.down_sample))
            # [batch_size, num_relation, down_sample]
            hyperedges = self.trans(hyperedges, edge_types)

            key2 = self.K2(hyperedges)
            key2 = key2.reshape(batch_size, -1, self.num_heads, self.head_size)

            value2 = self.V2(hyperedges)
            value2 = value2.reshape(batch_size, -1, self.num_heads, self.head_size)

            query2 = self.Q2(x)
            query2 = torch.gather(query2, 1, hyperedge_indexs_s.unsqueeze(-1).expand(-1, -1, self.down_sample))
            query2 = query2.reshape(batch_size, -1, self.num_heads, self.head_size)
            # [batch_size, num_relation, num_heads, head_size]

            attn2 = (query2 * key2).sum(dim=-1)
            attn2 = attn2 / math.sqrt(self.head_size)
            # [batch_size, num_relation, num_heads]
            attn_score2 = softmax(attn2, hyperedge_indexs_s, x.size(1), self.torch_dtype)
            attn_score2 = attn_score2.unsqueeze(-1)
            # [batch_size, num_relation, num_heads, 1]

            agg2 = value2 * attn_score2
            agg2 = agg2.reshape(batch_size, -1, self.down_sample)
            # [batch_size, num_relation, down_sample]

            down2 = torch.zeros(batch_size, x.size(1), self.down_sample, dtype=self.torch_dtype,
                                device=x.device).scatter_add_ \
                (1, hyperedge_indexs_s.unsqueeze(-1).expand(-1, -1, self.down_sample), agg2)
            # [batch_size, seq_len+1, down_sample]

            if self.use_norm:
                down2 = self.layer_norm(down2)

            down2 = self.dropout(down2)

            down2 = self.non_linearity(down2)
            # new add end
            up2 = self.feedforward_up(down2)

            if source_mask is not None:
                up2 = up2.masked_fill_(~source_mask, 0)

        elif self.use_adapter2 and source_mask is not None:
            down2 = self.feedforward_down(x)
            down2 = self.non_linearity(down2)
            up2 = self.feedforward_up(down2)
            up2 = up2.masked_fill_(~source_mask, 0)

        else:
            up2 = 0

        up = up1 + up2

        up = up * self.scaling
        output = up

        if self.use_gating:
            # x.shape = (batch_size, seq_len, hidden_size)
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            output = output * gate

        # apply residual connection before layer norm if configured in this way
        if self.adapter_residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        # if residual should be applied after layer norm, apply it here
        if not self.adapter_residual_before_ln:
            output = output + residual_input

        if self.use_gating and output_gating:
            return output, down, up, gate
        return output, down, up