from typing import Mapping

import torch
from torch import nn

from adapters import ForwardContext

from adapters.composition import adjust_tensors_for_parallel, Stack, Fuse, Split, Parallel, BatchSplit, \
    AdapterCompositionBlock, Average
from adapters.methods.bottleneck import BottleneckLayer
from adapters.methods.modeling import ParallelAdapter

from .HGAdapter import HGAdapter
from .HGAdapterConfig import HGAdapterConfig


# copied from AdapterLayer, only modify part of calling HGAdapter
class HGAdapterLayer(BottleneckLayer):
    def __init__(self, location_key: str):
        super().__init__(location_key)

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        adapter_config = self.adapters_config.match(
            adapter_name,
            config_type=HGAdapterConfig,
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
                adapter_class = ParallelAdapter
            else:
                adapter_class = HGAdapter
            adapter = adapter_class(
                adapter_name=adapter_name,
                input_size=self.model_config.hidden_size,
                down_sample=int(self.model_config.hidden_size // reduction_factor),
                config=adapter_config,
            )
            adapter.train(self.training)  # make sure training mode is consistent
            self.adapters[adapter_name] = adapter
            return True

        return False

    def compose_hg_stack(self, adapter_setup: Stack, hidden_states, residual_input, layer_norm, hyperedge_indexs,
                         edge_types, o,
                         lvl: int = 0):
        last = adapter_setup[0]
        for i, adapter_stack_layer in enumerate(adapter_setup):
            if adapter_stack_layer in self.adapter_modules:
                first_adapter = self.adapters[adapter_stack_layer]
                hidden_states, _, adapter_residual_input = first_adapter.pre_forward(
                    hidden_states, residual_input, layer_norm
                )
                hidden_states, up, o, last = self.compose_hg_single(adapter_stack_layer, hidden_states,
                                                                    adapter_residual_input, hyperedge_indexs,
                                                                    edge_types, o, lvl=lvl + 1)
            else:
                pass

        return hidden_states, o, last

    def compose_hg_single(self, adapter_setup, hidden_states, residual_input, hyperedge_indexs, edge_types, o,
                          lvl: int = 0):
        adapter_layer = self.adapters[adapter_setup]
        context = ForwardContext.get_context()
        output_gating = context.output_adapter_gating_scores if context is not None else False
        layer_output = adapter_layer(
            hidden_states,
            residual_input=residual_input,
            output_gating=output_gating,
            hyperedge_indexs=hyperedge_indexs,
            edge_types=edge_types,
            o=o
        )
        hidden_states, up = layer_output[0], layer_output[2]
        o = layer_output[3]
        if output_gating:
            self._store_gating_score(adapter_setup, layer_output[-1])

        return hidden_states, up, o, adapter_setup

    def adapter_layer_forward(self, hidden_states, residual_input, layer_norm, hyperedge_indexs=None, edge_types=None,
                              o=None):
        (residual_input,) = adjust_tensors_for_parallel(hidden_states, residual_input)
        # Replicate in both directions as residual might be larger (e.g. GPT-J)
        (hidden_states,) = adjust_tensors_for_parallel(residual_input, hidden_states)
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None:
            input_hidden_states = hidden_states

            hidden_states, o, last = self.compose_hg_stack(adapter_setup, hidden_states, residual_input,
                                                           layer_norm, hyperedge_indexs, edge_types, o)

            last_adapter = self.adapters[last]
            hidden_states = last_adapter.post_forward(hidden_states, input_hidden_states, residual_input, layer_norm)

        elif layer_norm:
            hidden_states = layer_norm(hidden_states + residual_input)
        else:
            hidden_states = hidden_states + residual_input

        return hidden_states, o

    def forward(self, hidden_states, residual_input, layer_norm, hyperedge_indexs=None, edge_types=None, o=None):
        return self.adapter_layer_forward(hidden_states, residual_input, layer_norm, hyperedge_indexs, edge_types, o)
