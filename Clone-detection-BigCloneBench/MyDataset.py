import json
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.nn.functional import pad
from tqdm import tqdm

import sys

sys.path.append("..")
from utils import pre_walk_tree_sitter_with_hyper


class CloneDetectionBCBHyperDataset(Dataset):
    def __init__(self, data_path, src_data_path, idx_file_path, tokenizer, parser, max_seq_len=512, overwrite=False):
        self.data_path = data_path
        self.idx = []

        # process
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        print("start making dataset")
        src_data = open(src_data_path).readlines()

        with tqdm(total=len(src_data)) as bar:
            for line in src_data:

                line = line.strip()
                line_data = json.loads(line)

                processed_data_path = os.path.join(self.data_path, line_data["idx"])
                if not os.path.exists(processed_data_path) or overwrite:
                    if not os.path.exists(processed_data_path):
                        os.makedirs(processed_data_path)
                    code = line_data["func"]
                    tree = parser.parse(bytes(code, "utf8"))
                    tokens, hyperedge_indexs, edge_types = pre_walk_tree_sitter_with_hyper(tree.root_node, 0, 0,
                                                                                           tokenizer)

                    code = ' '.join(code.split())
                    ref_tokens = tokenizer.tokenize(code)
                    ref = set(ref_tokens)

                    for i in range(len(tokens)):
                        if 'Ġ' + tokens[i] in ref:
                            tokens[i] = 'Ġ' + tokens[i]

                    tokens = [tokenizer.cls_token] + tokens[:max_seq_len - 2] + [tokenizer.sep_token]
                    code_input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    input = torch.tensor(code_input_ids, dtype=torch.long)

                    cut_hyperedge_indexs = [[], []]
                    cut_edge_types = []
                    for hyperedge_index_s, hyperedge_index_t, edge_type in zip(hyperedge_indexs[0], hyperedge_indexs[1],
                                                                               edge_types):
                        if (hyperedge_index_s + 1) < max_seq_len - 1:
                            cut_hyperedge_indexs[0].append(hyperedge_index_s + 1)
                            cut_hyperedge_indexs[1].append(hyperedge_index_t)
                            cut_edge_types.append(edge_type)

                    cut_hyperedge_indexs = torch.tensor(cut_hyperedge_indexs, dtype=torch.long)
                    cut_edge_types = torch.tensor(cut_edge_types, dtype=torch.long)

                    torch.save(input, os.path.join(processed_data_path, "input.pt"))
                    torch.save(cut_hyperedge_indexs, os.path.join(processed_data_path, "hyperedge_indexs.pt"))
                    torch.save(cut_edge_types, os.path.join(processed_data_path, "edge_types.pt"))

                bar.update(1)

        idx_file = open(idx_file_path)
        for line in idx_file:
            line = line.strip()
            idx1, idx2, label = line.split('\t')
            label = int(label)
            self.idx.append([idx1, idx2, label])
        idx_file.close()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        input1 = torch.load(os.path.join(self.data_path, self.idx[item][0], "input.pt"), weights_only=True)
        input2 = torch.load(os.path.join(self.data_path, self.idx[item][1], "input.pt"), weights_only=True)
        hyperedge_indexs1 = torch.load(os.path.join(self.data_path, self.idx[item][0], "hyperedge_indexs.pt"), weights_only=True)
        hyperedge_indexs2 = torch.load(os.path.join(self.data_path, self.idx[item][1], "hyperedge_indexs.pt"), weights_only=True)
        edge_types1 = torch.load(os.path.join(self.data_path, self.idx[item][0], "edge_types.pt"), weights_only=True)
        edge_types2 = torch.load(os.path.join(self.data_path, self.idx[item][1], "edge_types.pt"), weights_only=True)

        label = torch.tensor([self.idx[item][2]], dtype=torch.long)
        return input1, input2, hyperedge_indexs1, hyperedge_indexs2, edge_types1, edge_types2, label


class CloneDetectionBCBHyperCollater:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        inputs = []
        hyperedge_indexs = []
        edge_types = []
        labels = []
        for row in data:
            inputs.append(row[0])
            inputs.append(row[1])
            hyperedge_indexs.append(row[2])
            hyperedge_indexs.append(row[3])
            edge_types.append(row[4])
            edge_types.append(row[5])
            labels.append(row[6])

        labels = torch.cat(labels, dim=0)
        inputs = self.tokenizer.pad({"input_ids": inputs})
        padded_inputs = inputs["input_ids"]
        padded_inputs = pad(padded_inputs, [0, 1, 0, 0], mode="constant", value=self.tokenizer.pad_token_id)

        # False - is pad
        attention_masks = inputs["attention_mask"]
        attention_masks = pad(attention_masks, [0, 1, 0, 0], mode="constant", value=0)

        padded_hyperedge_indexs = []
        pad_length = max([hyperedge_index.size(1) for hyperedge_index in hyperedge_indexs])
        max_edge_plus1 = int(max([hyperedge_index[1, :].max() for hyperedge_index in hyperedge_indexs])) + 1
        for i in range(len(hyperedge_indexs)):
            hyperedge_index = hyperedge_indexs[i]
            padded_hyperedge_index = torch.zeros(2, pad_length, dtype=torch.long)
            padded_hyperedge_index[0] = pad(hyperedge_index[0], [0, pad_length - hyperedge_index.size(1)],
                                            mode="constant", value=padded_inputs.size(1) - 1)
            padded_hyperedge_index[1] = pad(hyperedge_index[1], [0, pad_length - hyperedge_index.size(1)],
                                            mode="constant", value=max_edge_plus1)

            padded_hyperedge_index = padded_hyperedge_index.unsqueeze(0)
            padded_hyperedge_indexs.append(padded_hyperedge_index)
        padded_hyperedge_indexs = torch.cat(padded_hyperedge_indexs, dim=0)

        padded_edge_types = pad_sequence(edge_types, batch_first=True, padding_value=0)

        return padded_inputs, attention_masks, padded_hyperedge_indexs, padded_edge_types, labels
