from typing import Mapping

import torch
from adapters import ForwardContext

from adapters.composition import adjust_tensors_for_parallel, Stack, Fuse, Split, Parallel, BatchSplit, \
    AdapterCompositionBlock, Average
from adapters.methods.bottleneck import BottleneckLayer

from .HyperStructAdapter import HyperStructAdapter, ParallelHyperStructAdapter, HGbdAdapter
from .HyperStructAdapterConfig import HyperStructAdapterConfig


# copied from AdapterLayer, only modify part of calling HyperStructAdapter
class HyperStructAdapterLayer(BottleneckLayer):
    def __init__(self, location_key: str):
        super().__init__(location_key)

    def add_adapter(self, adapter_name: str, layer_idx: int):
        self.layer_idx = layer_idx
        adapter_config = self.adapters_config.match(
            adapter_name,
            config_type=HyperStructAdapterConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if adapter_config is not None:
            reduction_factor = adapter_config["reduction_factor"]
            if isinstance(reduction_factor, Mapping):
                if str(self.layer_idx) in reduction_factor:
                    reduction_factor = reduction_factor[str(self.layer_idx)]
                elif "default" in reduction_factor:
                    reduction_factor = reduction_factor["default"]
                else:
                    raise KeyError(
                        "The given reduction factor mapping does not give a default value and does not specify each "
                        "reduction factor individually. You need to provide a default value like this: "
                        '{"1": 16, "default": 16}'
                    )

            if adapter_config.is_parallel:
                adapter_class = ParallelHyperStructAdapter
            elif adapter_config.is_newHGbd:
                adapter_class = HGbdAdapter
            else:
                adapter_class = HyperStructAdapter
            adapter = adapter_class(
                adapter_name=adapter_name,
                input_size=self.model_config.hidden_size,
                down_sample=int(self.model_config.hidden_size // reduction_factor),
                config=adapter_config,
            )
            adapter.train(self.training)
            self.adapters[adapter_name] = adapter
            return True

        return False

    def adapter_stack(self, adapter_setup: Stack, hidden_states, input_tensor, layer_norm, lvl=0,
                      hyperedge_indexs=None, edge_types=None):
        """
        Forwards the given input through the given stack of adapters.
        """
        for i, adapter_stack_layer in enumerate(adapter_setup):
            # Break if setup is too deep
            if isinstance(adapter_stack_layer, AdapterCompositionBlock) and lvl >= 1:
                raise ValueError(
                    "Specified adapter setup is too deep. Cannot have {} at level {}".format(
                        adapter_stack_layer.__class__.__name__, lvl
                    )
                )
            # Case 1: We have a nested fusion layer -> call fusion method
            if isinstance(adapter_stack_layer, Fuse):
                hidden_states = self.adapter_fusion(
                    adapter_stack_layer, hidden_states, input_tensor, layer_norm, lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types
                )
            # Case 2: We have a nested split layer -> call split method
            elif isinstance(adapter_stack_layer, Split):
                hidden_states = self.adapter_split(
                    adapter_stack_layer, hidden_states, input_tensor, layer_norm, lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types
                )
            # Case 3: We have a nested parallel layer -> call parallel method
            elif isinstance(adapter_stack_layer, Parallel):
                hidden_states, input_tensor = self.adapter_parallel(
                    adapter_stack_layer, hidden_states, input_tensor, layer_norm, lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types
                )
            # Case 4: We have a nested batch split block -> call batchsplit method
            elif isinstance(adapter_stack_layer, BatchSplit):
                hidden_states = self.adapter_batchsplit(
                    adapter_stack_layer, hidden_states, input_tensor, layer_norm, lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types
                )
            # Case 5: We have a nested average block -> call average method
            elif isinstance(adapter_stack_layer, Average):
                hidden_states = self.adapter_average_output(
                    adapter_stack_layer, hidden_states, input_tensor, layer_norm, lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types
                )
            # Case 6: We have a single adapter which is part of this module -> forward pass
            elif adapter_stack_layer in self.adapters:
                adapter_layer = self.adapters[adapter_stack_layer]
                hidden_states, _, residual = adapter_layer.pre_forward(hidden_states, input_tensor, layer_norm)
                context = ForwardContext.get_context()
                layer_output = adapter_layer(
                    hidden_states, residual_input=residual, output_gating=context.output_adapter_gating_scores,
                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types
                )
                hidden_states, up = layer_output[0], layer_output[2]
                self._store_gating_score(adapter_stack_layer, layer_output[-1])
                # as this stack might be part of a fusion block, return the adapter up-projection output here
                # together with the final output (with potential residuals & norms) if we reached the last block of the stack
                if i == len(adapter_setup) - 1:
                    return hidden_states, up, input_tensor
            # Case X: No adapter which is part of this module -> ignore

        # If we got here, we either had another nested composition block
        # or no adapter was found. In both cases, we don't need to set the second return value for fusion
        return hidden_states, None, input_tensor

    def adapter_fusion(self, adapter_setup: Fuse, hidden_states, input_tensor, layer_norm, lvl=0,
                       hyperedge_indexs=None, edge_types=None):
        """
        Performs adapter fusion with the given adapters for the given input.
        """
        context = ForwardContext.get_context()

        # config of _last_ fused adapter is significant
        fusion_config = self.config.adapters.get_fusion(adapter_setup.name)
        last_adapter = self.adapters[adapter_setup.last()]
        hidden_states, query, residual = last_adapter.pre_forward(
            hidden_states, input_tensor, layer_norm, fusion_config=fusion_config
        )

        up_list = []

        for adapter_block in adapter_setup:
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                _, up, _ = self.adapter_stack(adapter_block, hidden_states, input_tensor, layer_norm, lvl=lvl + 1,
                                              hyperedge_indexs=hyperedge_indexs, edge_types=edge_types)
                if up is not None:  # could be none if stack is empty
                    up_list.append(up)
            # Case 2: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.adapters:
                adapter_layer = self.adapters[adapter_block]
                layer_output = adapter_layer(
                    hidden_states, residual_input=residual, output_gating=context.output_adapter_gating_scores,
                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types
                )
                up = layer_output[2]
                self._store_gating_score(adapter_block, layer_output[-1])
                up_list.append(up)
            # Case 3: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore

        if len(up_list) > 0:
            up_list = torch.stack(up_list)
            up_list = up_list.permute(1, 2, 0, 3)

            fusion_output = self.adapter_fusion_layer[adapter_setup.name](
                query,
                up_list,
                up_list,
                residual,
                output_attentions=context.output_adapter_fusion_attentions,
            )
            if context.output_adapter_fusion_attentions:
                hidden_states = fusion_output[0]
                self._store_fusion_attentions(adapter_setup.name, fusion_output[-1])
            else:
                hidden_states = fusion_output

        return hidden_states

    def adapter_split(self, adapter_setup: Split, hidden_states, input_tensor, layer_norm, lvl=0,
                      hyperedge_indexs=None, edge_types=None):
        """
        Splits the given input between the given adapters.
        """
        # config of _first_ of split adapters is significant
        first_adapter = self.adapters[adapter_setup.first()]
        hidden_states, query, residual = first_adapter.pre_forward(hidden_states, input_tensor, layer_norm)

        # split hidden representations and residuals at split index
        split_hidden_states = [
            hidden_states[:, : adapter_setup.split_index, :],
            hidden_states[:, adapter_setup.split_index:, :],
        ]
        split_input_tensor = [
            input_tensor[:, : adapter_setup.split_index, :],
            input_tensor[:, adapter_setup.split_index:, :],
        ]
        split_residual = [
            residual[:, : adapter_setup.split_index, :],
            residual[:, adapter_setup.split_index:, :],
        ]

        for i, adapter_block in enumerate(adapter_setup):
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                split_hidden_states[i], _, _ = self.adapter_stack(
                    adapter_block, split_hidden_states[i], split_input_tensor[i], layer_norm, lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types
                )
            # Case 2: We have a nested split -> recursively call split
            elif isinstance(adapter_block, Split):
                split_hidden_states[i] = self.adapter_split(
                    adapter_block, split_hidden_states[i], split_input_tensor[i], layer_norm, lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types
                )
            # Case 3: We have a nested batch split -> call batch split method
            elif isinstance(adapter_block, BatchSplit):
                split_hidden_states[i] = self.adapter_batchsplit(
                    adapter_block, split_hidden_states[i], split_input_tensor[i], layer_norm, lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types
                )
            # Case 4: We have a nested average -> call average method
            elif isinstance(adapter_block, Average):
                split_hidden_states[i] = self.adapter_average_output(
                    adapter_block, split_hidden_states[i], split_input_tensor[i], layer_norm, lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types
                )
            # Case 5: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.adapters:
                adapter_layer = self.adapters[adapter_block]
                context = ForwardContext.get_context()
                layer_output = adapter_layer(
                    split_hidden_states[i],
                    residual_input=split_residual[i],
                    output_gating=context.output_adapter_gating_scores,
                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types
                )
                split_hidden_states[i] = layer_output[0]
                self._store_gating_score(adapter_block, layer_output[-1])
            # Case 6: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore

        hidden_states = torch.cat(split_hidden_states, dim=1)
        return hidden_states

    def adapter_parallel(self, adapter_setup: Parallel, hidden_states, input_tensor, layer_norm, lvl=0,
                         hyperedge_indexs=None, edge_types=None):
        """
        For parallel execution of the adapters on the same input. This means that the input is repeated N times before
        feeding it to the adapters (where N is the number of adapters).
        """

        context = ForwardContext.get_context()
        if not context.adapters_parallelized:
            orig_batch_size = input_tensor.shape[0]
            input_tensor = input_tensor.repeat(self.config.adapters.active_setup.parallel_channels, 1, 1)
            hidden_states = hidden_states.repeat(self.config.adapters.active_setup.parallel_channels, 1, 1)
            context.adapters_parallelized = True
        else:
            # The base model should handle replication of input.
            # Therefore, we assume the (replicated) input batch to be divisible by the number of parallel channels.
            if hidden_states.shape[0] % adapter_setup.parallel_channels != 0:
                raise ValueError(
                    "The total input batch size in a Parallel adapter block must be divisible by the number of"
                    " parallel channels."
                )
            orig_batch_size = hidden_states.shape[0] // adapter_setup.parallel_channels

        # We assume all adapters have the same config
        first_adapter = self.adapters[adapter_setup.first()]
        hidden_states, _, residual = first_adapter.pre_forward(hidden_states, input_tensor, layer_norm)

        # sequentially feed different parts of the blown-up batch into different adapters
        children_hidden = []
        for i, child in enumerate(adapter_setup):
            # Case 1: We have a nested stack -> call stack method
            if isinstance(child, Stack):
                child_hidden_states, _, _ = self.adapter_stack(
                    child,
                    hidden_states[i * orig_batch_size: (i + 1) * orig_batch_size],
                    input_tensor[i * orig_batch_size: (i + 1) * orig_batch_size],
                    layer_norm,
                    lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs,
                    edge_types=edge_types
                )
                children_hidden.append(child_hidden_states)
            # Case 2. We have a nested batchsplit block -> call batchsplit method
            elif isinstance(child, BatchSplit):
                child_hidden_states = self.adapter_batchsplit(
                    child,
                    hidden_states[i * orig_batch_size: (i + 1) * orig_batch_size],
                    input_tensor[i * orig_batch_size: (i + 1) * orig_batch_size],
                    layer_norm,
                    lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs,
                    edge_types=edge_types
                )
                children_hidden.append(child_hidden_states)
            # Case 3: We have a nested average block -> call average method
            elif isinstance(child, Average):
                child_hidden_states = self.adapter_average_output(
                    child,
                    hidden_states[i * orig_batch_size: (i + 1) * orig_batch_size],
                    input_tensor[i * orig_batch_size: (i + 1) * orig_batch_size],
                    layer_norm,
                    lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs,
                    edge_types=edge_types
                )
                children_hidden.append(child_hidden_states)
            # Case 4: We have a single adapter which is part of this module -> forward pass
            elif child in self.adapters:
                adapter_layer = self.adapters[child]
                context = ForwardContext.get_context()
                layer_output = adapter_layer(
                    hidden_states[i * orig_batch_size: (i + 1) * orig_batch_size],
                    residual_input=residual[i * orig_batch_size: (i + 1) * orig_batch_size],
                    output_gating=context.output_adapter_gating_scores,
                    hyperedge_indexs=hyperedge_indexs,
                    edge_types=edge_types
                )
                child_hidden_states = layer_output[0]
                self._store_gating_score(child, layer_output[-1])
                children_hidden.append(child_hidden_states)
            # Case 5: nesting other composition blocks is invalid
            elif isinstance(child, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        child.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore
            else:
                children_hidden.append(hidden_states[i * orig_batch_size: (i + 1) * orig_batch_size])

        # concatenate all outputs and return
        hidden_states = torch.cat(children_hidden, 0)
        return hidden_states, input_tensor

    def adapter_batchsplit(self, adapter_setup: BatchSplit, hidden_states, input_tensor, layer_norm, lvl=0,
                           hyperedge_indexs=None, edge_types=None):
        if not sum(adapter_setup.batch_sizes) == hidden_states.shape[0]:
            raise IndexError(
                "The given batch has a size of {} which is not compatible with batch_sizes {}".format(
                    hidden_states.shape[0], adapter_setup.batch_sizes
                )
            )

        first_adapter = self.adapters[adapter_setup.first()]
        hidden_states, _, residual = first_adapter.pre_forward(hidden_states, input_tensor, layer_norm)
        children_hidden = []
        for i, adapter_block in enumerate(adapter_setup):
            # compute ids of sequences thet should be passed to the ith adapter
            batch_idx = (
                sum(adapter_setup.batch_sizes[:i]),
                sum(adapter_setup.batch_sizes[: i + 1]),
            )
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                child, _, _ = self.adapter_stack(
                    adapter_block,
                    hidden_states[batch_idx[0]: batch_idx[1]],
                    input_tensor[batch_idx[0]: batch_idx[1]],
                    layer_norm,
                    lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs,
                    edge_types=edge_types
                )
                children_hidden.append(child)
            # Case 2: We have a nested split -> recursively call split
            elif isinstance(adapter_block, Split):
                child = self.adapter_split(
                    adapter_block,
                    hidden_states[batch_idx[0]: batch_idx[1]],
                    input_tensor[batch_idx[0]: batch_idx[1]],
                    layer_norm,
                    lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs,
                    edge_types=edge_types
                )
                children_hidden.append(child)
            # Case 3: We have a nested batch split block -> call batchsplit method
            elif isinstance(adapter_block, BatchSplit):
                child = self.adapter_batchsplit(
                    adapter_block,
                    hidden_states[batch_idx[0]: batch_idx[1]],
                    input_tensor[batch_idx[0]: batch_idx[1]],
                    layer_norm,
                    lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs,
                    edge_types=edge_types
                )
                children_hidden.append(child)
            # Case 4: We have a nested average block -> call average method
            elif isinstance(adapter_block, Average):
                child = self.adapter_average_output(
                    adapter_block,
                    hidden_states[batch_idx[0]: batch_idx[1]],
                    input_tensor[batch_idx[0]: batch_idx[1]],
                    layer_norm,
                    lvl=lvl + 1,
                    hyperedge_indexs=hyperedge_indexs,
                    edge_types=edge_types
                )
                children_hidden.append(child)
            # Case 5: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.adapters:

                adapter_layer = self.adapters[adapter_block]
                context = ForwardContext.get_context()
                layer_output = adapter_layer(
                    hidden_states[batch_idx[0]: batch_idx[1]],
                    residual_input=residual[batch_idx[0]: batch_idx[1]],
                    output_gating=context.output_adapter_gating_scores,
                    hyperedge_indexs=hyperedge_indexs,
                    edge_types=edge_types
                )
                children_hidden.append(layer_output[0])
                self._store_gating_score(adapter_block, layer_output[-1])
            # Case 6: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore
            else:
                children_hidden.append(hidden_states[batch_idx])

        hidden_states = torch.cat(children_hidden, 0)
        return hidden_states

    def adapter_average_output(self, adapter_setup: Average, hidden_states, input_tensor, layer_norm, lvl=0,
                               hyperedge_indexs=None, edge_types=None):
        """
        For averaging the output representations of multiple adapters.
        """
        context = ForwardContext.get_context()

        # We assume all adapters have the same config
        first_adapter = self.adapters[adapter_setup.first()]
        hidden_states, _, residual = first_adapter.pre_forward(hidden_states, input_tensor, layer_norm)

        children_hidden = []

        for adapter_block in adapter_setup:
            # Case 1: We have a nested stack -> call stack method
            if isinstance(adapter_block, Stack):
                child, _, _ = self.adapter_stack(adapter_block, hidden_states, input_tensor, layer_norm, lvl=lvl + 1,
                                                 hyperedge_indexs=hyperedge_indexs, edge_types=edge_types)
                children_hidden.append(child)
            # Case 2: We have a nested split block -> call split method
            elif isinstance(adapter_block, Split):
                child = self.adapter_split(adapter_block, hidden_states, input_tensor, layer_norm, lvl=lvl + 1,
                                           hyperedge_indexs=hyperedge_indexs, edge_types=edge_types)
                children_hidden.append(child)
            # Case 3: We have a nested batch split block -> call batchsplit method
            elif isinstance(adapter_block, BatchSplit):
                child = self.adapter_batchsplit(adapter_block, hidden_states, input_tensor, layer_norm, lvl=lvl + 1,
                                                hyperedge_indexs=hyperedge_indexs, edge_types=edge_types)
                children_hidden.append(child)
            # Case 4: We have a single adapter which is part of this module -> forward pass
            elif adapter_block in self.adapters:
                adapter_layer = self.adapters[adapter_block]
                layer_output = adapter_layer(
                    hidden_states, residual_input=residual, output_gating=context.output_adapter_gating_scores,
                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types
                )
                children_hidden.append(layer_output[0])
                self._store_gating_score(adapter_block, layer_output[-1])
            # Case 5: nesting other composition blocks is invalid
            elif isinstance(adapter_block, AdapterCompositionBlock):
                raise ValueError(
                    "Invalid adapter setup. Cannot nest {} in {}".format(
                        adapter_block.__class__.__name__, adapter_setup.__class__.__name__
                    )
                )
            # Case X: No adapter which is part of this module -> ignore

        weights = torch.tensor(adapter_setup.weights).unsqueeze(1).unsqueeze(1).to(hidden_states.device)
        hidden_states = torch.mean(torch.cat(children_hidden, 0) * weights, 0)

        return hidden_states

    def adapter_layer_forward(self, hidden_states, residual_input, layer_norm, hyperedge_indexs=None, edge_types=None):
        (residual_input,) = adjust_tensors_for_parallel(hidden_states, residual_input)
        (hidden_states,) = adjust_tensors_for_parallel(residual_input, hidden_states)
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None:
            input_hidden_states = hidden_states

            if isinstance(adapter_setup, Stack):
                hidden_states, _, residual_input = self.adapter_stack(
                    adapter_setup, hidden_states, residual_input, layer_norm, hyperedge_indexs=hyperedge_indexs,
                    edge_types=edge_types
                )
            elif isinstance(adapter_setup, Fuse):
                hidden_states = self.adapter_fusion(adapter_setup, hidden_states, residual_input, layer_norm,
                                                    hyperedge_indexs=hyperedge_indexs, edge_types=edge_types)
            elif isinstance(adapter_setup, Split):
                hidden_states = self.adapter_split(adapter_setup, hidden_states, residual_input, layer_norm,
                                                   hyperedge_indexs=hyperedge_indexs, edge_types=edge_types)
            elif isinstance(adapter_setup, Parallel):
                hidden_states, residual_input = self.adapter_parallel(
                    adapter_setup, hidden_states, residual_input, layer_norm, hyperedge_indexs=hyperedge_indexs,
                    edge_types=edge_types
                )
            elif isinstance(adapter_setup, BatchSplit):
                hidden_states = self.adapter_batchsplit(adapter_setup, hidden_states, residual_input, layer_norm,
                                                        hyperedge_indexs=hyperedge_indexs, edge_types=edge_types)
            elif isinstance(adapter_setup, Average):
                hidden_states = self.adapter_average_output(adapter_setup, hidden_states, residual_input, layer_norm,
                                                            hyperedge_indexs=hyperedge_indexs, edge_types=edge_types)
            else:
                raise ValueError(f"Invalid adapter setup {adapter_setup}")

            last_adapter = self.adapters[adapter_setup.last()]
            hidden_states = last_adapter.post_forward(hidden_states, input_hidden_states, residual_input, layer_norm)

        elif layer_norm:
            hidden_states = layer_norm(hidden_states + residual_input)
        else:
            hidden_states = hidden_states + residual_input

        return hidden_states

    def forward(self, hidden_states, residual_input, layer_norm, hyperedge_indexs=None, edge_types=None):
        return self.adapter_layer_forward(hidden_states, residual_input, layer_norm, hyperedge_indexs, edge_types)
