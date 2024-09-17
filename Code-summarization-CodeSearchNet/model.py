from transformers import EncoderDecoderModel


class MyHyperEncoderDecoderModel(EncoderDecoderModel):
    # only overwrite this, the same as others
    def prepare_inputs_for_generation(
            self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None,
            hyperedge_indexs=None, edge_types=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
            "hyperedge_indexs": hyperedge_indexs,
            "edge_types": edge_types
        }
        return input_dict
