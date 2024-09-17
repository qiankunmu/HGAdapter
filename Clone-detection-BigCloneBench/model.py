import torch
import torch.nn as nn
from torch.nn.functional import relu

class CloneDetectionBCBHyperModel(nn.Module):
    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.feed_forward = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, hyperedge_indexs=None, edge_types=None):
        output = self.model(input_ids, attention_mask, hyperedge_indexs=hyperedge_indexs,
                            edge_types=edge_types).last_hidden_state
        # [batch_size * 2, seq_len, hidden_size]
        x = output[0:, 0, 0:]
        # [batch_size * 2, hidden_size]
        x = x.reshape(-1, x.size(-1) * 2)
        # [batch_size, hidden_size * 2]
        x = self.feed_forward(x)
        x = relu(x)
        x = self.dropout(x)
        logits = self.output_layer(x)
        # [batch_size, 2]
        return logits

