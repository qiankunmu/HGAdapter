from adapters.methods.lora import LoRALinear

from .HGAdapterLayer import HGAdapterLayer


class LlamaDecoderLayerHGAdaptersMixin:
    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.mlp.down_proj = LoRALinear.wrap(self.mlp.down_proj, "intermediate", model_config, adapters_config)
        self.mlp.up_proj = LoRALinear.wrap(self.mlp.up_proj, "output", model_config, adapters_config)

        self.attention_adapters = HGAdapterLayer("mh_adapter")
        self.output_adapters = HGAdapterLayer("output_adapter")