from .HGAdapterLayer import HGAdapterLayer

# For backwards compatibility, BertSelfOutput inherits directly from AdapterLayer
class BertSelfOutputHGAdaptersMixin(HGAdapterLayer):
    """Adds adapters to the BertSelfOutput module."""

    def __init__(self):
        super().__init__("mh_adapter")

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "mh_adapter"
        super().init_adapters(model_config, adapters_config)


# For backwards compatibility, BertOutput inherits directly from AdapterLayer
class BertOutputHGAdaptersMixin(HGAdapterLayer):
    """Adds adapters to the BertOutput module."""

    def __init__(self):
        super().__init__("output_adapter")

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "output_adapter"
        super().init_adapters(model_config, adapters_config)
