import json
import os

import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
import re

import sys

sys.path.append("..")
from utils import pre_walk_tree_sitter_with_hyper


class CodeSummarizationCSNHyperDataset(Dataset):
    def __init__(self, data_path, src_data_path, tokenizer, parser, max_seq_len=512, max_tgt_len=128, overwrite=False):
        self.data_path = data_path

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        print("start making dataset")

        src_data = open(src_data_path).readlines()
        self.length = 0

        with tqdm(total=len(src_data)) as bar:
            for i, line in enumerate(src_data):

                line = line.strip()
                line_data = json.loads(line)
                code = line_data["code"]

                if line_data["language"] == "python":
                    code = re.sub(r'\'\'\'.*?\'\'\'|\"\"\".*?\"\"\"', '', code, count=1, flags=re.DOTALL)

                if len(code) > 300000:
                    continue

                processed_data_path = os.path.join(self.data_path, str(self.length))

                if not os.path.exists(processed_data_path) or overwrite:
                    if not os.path.exists(processed_data_path):
                        os.makedirs(processed_data_path)

                    tree = parser.parse(bytes(code, "utf8"))
                    source_tokens, hyperedge_indexs, edge_types = pre_walk_tree_sitter_with_hyper(tree.root_node, 0, 0,
                                                                                                  tokenizer)

                    code = ' '.join(line_data['code_tokens']).replace('\n', ' ')
                    code = ' '.join(code.strip().split())
                    code = ' '.join(code.split())
                    ref_tokens = tokenizer.tokenize(code)
                    ref = set(ref_tokens)

                    for i in range(len(source_tokens)):
                        if 'Ġ' + source_tokens[i] in ref:
                            source_tokens[i] = 'Ġ' + source_tokens[i]

                    source_tokens = [tokenizer.cls_token] + source_tokens[:max_seq_len - 2] + [tokenizer.sep_token]
                    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
                    source = torch.tensor(source_ids, dtype=torch.long)

                    nl = ' '.join(line_data['docstring_tokens']).replace('\n', '')
                    nl = ' '.join(nl.strip().split())
                    target_tokens = tokenizer.tokenize(nl)
                    target_tokens = target_tokens[:max_tgt_len - 1] + [tokenizer.sep_token]
                    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
                    target = torch.tensor(target_ids, dtype=torch.long)

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

                    torch.save(source, os.path.join(processed_data_path, "source.pt"))
                    torch.save(target, os.path.join(processed_data_path, "target.pt"))
                    torch.save(cut_hyperedge_indexs, os.path.join(processed_data_path, "hyperedge_indexs.pt"))
                    torch.save(cut_edge_types, os.path.join(processed_data_path, "edge_types.pt"))

                self.length += 1

                bar.update(1)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        source = torch.load(os.path.join(self.data_path, str(item), "source.pt"), weights_only=True)
        target = torch.load(os.path.join(self.data_path, str(item), "target.pt"), weights_only=True)
        hyperedge_indexs = torch.load(os.path.join(self.data_path, str(item), "hyperedge_indexs.pt"), weights_only=True)
        edge_types = torch.load(os.path.join(self.data_path, str(item), "edge_types.pt"), weights_only=True)

        return source, target, hyperedge_indexs, edge_types


class CodeSummarizationCSNHyperCollater:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        sources = []
        targets = []
        hyperedge_indexs = []
        edge_types = []
        for row in data:
            sources.append(row[0])
            targets.append(row[1])
            hyperedge_indexs.append(row[2])
            edge_types.append(row[3])

        source_inputs = self.tokenizer.pad({"input_ids": sources})
        sources = source_inputs["input_ids"]
        sources = pad(sources, [0, 1, 0, 0], mode="constant", value=self.tokenizer.pad_token_id)
        # False - is pad
        source_attention_masks = source_inputs["attention_mask"]
        source_attention_masks = pad(source_attention_masks, [0, 1, 0, 0], mode="constant", value=0)

        targets = pad_sequence(targets, batch_first=True, padding_value=-100)

        padded_hyperedge_indexs = []
        pad_length = max([hyperedge_index.size(1) for hyperedge_index in hyperedge_indexs])
        max_edge_plus1 = int(max([hyperedge_index[1, :].max() for hyperedge_index in hyperedge_indexs])) + 1
        for i in range(len(hyperedge_indexs)):
            hyperedge_index = hyperedge_indexs[i]
            padded_hyperedge_index = torch.zeros(2, pad_length, dtype=torch.long)
            padded_hyperedge_index[0] = pad(hyperedge_index[0], [0, pad_length - hyperedge_index.size(1)],
                                            mode="constant", value=sources.size(1) - 1)
            padded_hyperedge_index[1] = pad(hyperedge_index[1], [0, pad_length - hyperedge_index.size(1)],
                                            mode="constant", value=max_edge_plus1)

            padded_hyperedge_index = padded_hyperedge_index.unsqueeze(0)
            padded_hyperedge_indexs.append(padded_hyperedge_index)
        padded_hyperedge_indexs = torch.cat(padded_hyperedge_indexs, dim=0)

        padded_edge_types = pad_sequence(edge_types, batch_first=True, padding_value=0)

        return sources, targets, source_attention_masks, padded_hyperedge_indexs, padded_edge_types


class CodeSummarizationCSNLlamaDataset(Dataset):
    def __init__(self, data_path, src_data_path, tokenizer, parser, max_seq_len=512, max_tgt_len=128, overwrite=False,
                 mode="train"):
        self.data_path = data_path

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        print("start making dataset")

        src_data = open(src_data_path).readlines()
        self.length = 0

        with tqdm(total=len(src_data)) as bar:
            for i, line in enumerate(src_data):

                line = line.strip()
                line_data = json.loads(line)
                code = line_data["code"]

                if line_data["language"] == "python":
                    code = re.sub(r'\'\'\'.*?\'\'\'|\"\"\".*?\"\"\"', '', code, count=1, flags=re.DOTALL)

                if len(code) > 300000:
                    continue

                processed_data_path = os.path.join(self.data_path, str(self.length))

                if not os.path.exists(processed_data_path) or overwrite:
                    if not os.path.exists(processed_data_path):
                        os.makedirs(processed_data_path)

                    tree = parser.parse(bytes(code, "utf8"))
                    source_tokens, hyperedge_indexs, edge_types = pre_walk_tree_sitter_with_hyper(tree.root_node, 0, 0,
                                                                                                  tokenizer)

                    source_tokens = [tokenizer.bos_token] + source_tokens[:max_seq_len - 2] + [tokenizer.eos_token]
                    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

                    nl = ' '.join(line_data['docstring_tokens']).replace('\n', '')
                    nl = ' '.join(nl.strip().split())
                    target_tokens = tokenizer.tokenize(nl)
                    target_tokens = target_tokens[:max_tgt_len - 2] + [tokenizer.eos_token]
                    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

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

                    if mode == "train":
                        input_ids = source_ids + [tokenizer.bos_token_id] + target_ids
                        input_ids = torch.tensor(input_ids, dtype=torch.long)
                        torch.save(input_ids, os.path.join(processed_data_path, "input_ids.pt"))
                        labels = [-100 for i in range(len(source_ids) + 1)] + target_ids
                        labels = torch.tensor(labels, dtype=torch.long)
                        torch.save(labels, os.path.join(processed_data_path, "labels.pt"))
                    else:
                        input_ids = source_ids + [tokenizer.bos_token_id]
                        input_ids = torch.tensor(input_ids, dtype=torch.long)
                        torch.save(input_ids, os.path.join(processed_data_path, "input_ids.pt"))
                        labels = target_ids
                        labels = torch.tensor(labels, dtype=torch.long)
                        torch.save(labels, os.path.join(processed_data_path, "labels.pt"))

                    torch.save(cut_hyperedge_indexs, os.path.join(processed_data_path, "hyperedge_indexs.pt"))
                    torch.save(cut_edge_types, os.path.join(processed_data_path, "edge_types.pt"))

                self.length += 1

                bar.update(1)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        input_ids = torch.load(os.path.join(self.data_path, str(item), "input_ids.pt"), weights_only=True)
        labels = torch.load(os.path.join(self.data_path, str(item), "labels.pt"), weights_only=True)
        hyperedge_indexs = torch.load(os.path.join(self.data_path, str(item), "hyperedge_indexs.pt"), weights_only=True)
        edge_types = torch.load(os.path.join(self.data_path, str(item), "edge_types.pt"), weights_only=True)

        return input_ids, labels, hyperedge_indexs, edge_types


class CodeSummarizationCSNLlamaCollater:
    def __init__(self, tokenizer, mode="train"):
        self.tokenizer = tokenizer
        self.mode = mode

    def __call__(self, data):
        input_ids = []
        labels = []
        hyperedge_indexs = []
        edge_types = []
        inputs_length = []
        for row in data:
            input_ids.append(row[0])
            labels.append(row[1])
            hyperedge_indexs.append(row[2])
            edge_types.append(row[3])
            inputs_length.append(row[0].size(0))

        inputs = self.tokenizer.pad({"input_ids": input_ids})
        input_ids = inputs["input_ids"]
        padded_input_ids = pad(input_ids, [1, 0, 0, 0], mode="constant", value=self.tokenizer.pad_token_id)
        # False - is pad
        attention_mask = inputs["attention_mask"]
        attention_mask = pad(attention_mask, [1, 0, 0, 0], mode="constant", value=0)

        label_inputs = self.tokenizer.pad({"input_ids": labels})
        labels = label_inputs["input_ids"]
        if self.mode == "train":
            padded_labels = pad(labels, [1, 0, 0, 0], mode="constant", value=self.tokenizer.pad_token_id)
            padded_labels.masked_fill_(attention_mask == 0, -100)
        else:
            padded_labels = labels

        padded_hyperedge_indexs = []
        pad_length = max([hyperedge_index.size(1) for hyperedge_index in hyperedge_indexs])
        max_edge_plus1 = int(max([hyperedge_index[1, :].max() for hyperedge_index in hyperedge_indexs])) + 1
        for i in range(len(hyperedge_indexs)):
            hyperedge_index = hyperedge_indexs[i]
            hyperedge_index[0] += padded_input_ids.size(1) - inputs_length[i]

            padded_hyperedge_index = torch.zeros(2, pad_length, dtype=torch.long)
            padded_hyperedge_index[0] = pad(hyperedge_index[0], [0, pad_length - hyperedge_index.size(1)],
                                            mode="constant", value=0)
            padded_hyperedge_index[1] = pad(hyperedge_index[1], [0, pad_length - hyperedge_index.size(1)],
                                            mode="constant", value=max_edge_plus1)

            padded_hyperedge_index = padded_hyperedge_index.unsqueeze(0)
            padded_hyperedge_indexs.append(padded_hyperedge_index)
        padded_hyperedge_indexs = torch.cat(padded_hyperedge_indexs, dim=0)

        padded_edge_types = pad_sequence(edge_types, batch_first=True, padding_value=0)

        return padded_input_ids, attention_mask, padded_hyperedge_indexs, padded_edge_types, padded_labels
