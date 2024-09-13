import json
import os

import numpy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.nn.functional import pad
from tqdm import tqdm

import sys

sys.path.append("..")
from utils import pre_walk_tree_sitter, pre_walk_tree_sitter_with_subtoken, pre_walk_tree_sitter_with_hyper


class CloneDetectionBCBDataset(Dataset):
    def __init__(self, data_path, src_data_path, idx_file_path, tokenizer, parser, max_seq_len=512, overwrite=False):
        self.data_path = data_path
        self.idx = []
        # self.idx2input = {}
        # self.idx2matrix_distances = {}

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
                    tokens = []
                    _, matrix_distances = pre_walk_tree_sitter_with_subtoken(tree.root_node, tokens, tokenizer)

                    code = ' '.join(code.split())
                    ref_tokens = tokenizer.tokenize(code)
                    ref = set(ref_tokens)

                    for i in range(len(tokens)):
                        if 'Ġ' + tokens[i] in ref:
                            tokens[i] = 'Ġ' + tokens[i]

                    tokens = [tokenizer.cls_token] + tokens[:max_seq_len - 2] + [tokenizer.sep_token]
                    code_input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    input = torch.tensor(code_input_ids, dtype=torch.long)

                    # padded_input = pad(input, (0, max_seq_len - input.size(0)), "constant", tokenizer.pad_token_id)

                    matrix_distances = [m[:max_seq_len - 2] for m in matrix_distances]
                    matrix_distances = matrix_distances[:max_seq_len - 2]
                    matrix_distances_top = [0]
                    matrix_distances_down = [max_seq_len + 1]
                    for m in matrix_distances:
                        m.insert(0, max_seq_len + 1)
                        m.append(max_seq_len + 1)
                        matrix_distances_top.append(max_seq_len + 1)
                        matrix_distances_down.append(max_seq_len + 1)
                    matrix_distances_top.append(max_seq_len + 1)
                    matrix_distances_down.append(0)
                    matrix_distances.insert(0, matrix_distances_top)
                    matrix_distances.append(matrix_distances_down)

                    matrix_distances = torch.tensor(matrix_distances, dtype=torch.long)
                    matrix_distances_np = matrix_distances.numpy()

                    # padded_matrix_distances = pad(matrix_distances, [0, max_seq_len - matrix_distances.size(1),
                    #                                                  0, max_seq_len - matrix_distances.size(0)],
                    #                               "constant", max_seq_len + 1)
                    # padded_matrix_distances[matrix_distances.size(0):, max_seq_len - 1] = 0

                    torch.save(input, os.path.join(processed_data_path, "input.pt"))
                    # torch.save(matrix_distances, os.path.join(processed_data_path, "matrix_distances.pt"))
                    numpy.savez_compressed(os.path.join(processed_data_path, "matrix_distances"),
                                           matrix_distances_np)

                    # self.idx2input[line_data["idx"]] = padded_input.unsqueeze(0)
                    # self.idx2matrix_distances[line_data["idx"]] = padded_matrix_distances.unsqueeze(0)
                # else:
                #     self.idx2input[line_data["idx"]] = torch.load(os.path.join(processed_data_path, "input.pt"))
                #     self.idx2matrix_distances[line_data["idx"]] = torch.load(
                #         os.path.join(processed_data_path, "matrix_distances.pt"))

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
        input1 = torch.load(os.path.join(self.data_path, self.idx[item][0], "input.pt"))
        input2 = torch.load(os.path.join(self.data_path, self.idx[item][1], "input.pt"))
        matrix_distances1_np = numpy.load(os.path.join(self.data_path, self.idx[item][0], "matrix_distances.npz"))[
            "arr_0"]
        matrix_distances1 = torch.from_numpy(matrix_distances1_np)
        matrix_distances2_np = numpy.load(os.path.join(self.data_path, self.idx[item][1], "matrix_distances.npz"))[
            "arr_0"]
        matrix_distances2 = torch.from_numpy(matrix_distances2_np)
        # matrix_distances1 = torch.load(
        #     os.path.join(self.data_path, self.idx[item][0], "matrix_distances.pt"))
        # matrix_distances2 = torch.load(
        #     os.path.join(self.data_path, self.idx[item][1], "matrix_distances.pt"))
        # input1 = self.idx2input[self.idx[item][0]]
        # input2 = self.idx2input[self.idx[item][1]]
        # matrix_distances1 = self.idx2matrix_distances[self.idx[item][0]]
        # matrix_distances2 = self.idx2matrix_distances[self.idx[item][1]]

        label = torch.tensor([self.idx[item][2]], dtype=torch.long)
        return input1, input2, matrix_distances1, matrix_distances2, label


class CloneDetectionBCBCollater:
    def __init__(self, tokenizer, struct_walk_size=2):
        self.tokenizer = tokenizer
        self.struct_walk_size = struct_walk_size

    def __call__(self, data):
        inputs = []
        struct_masks = []
        labels = []
        for row in data:
            inputs.append(row[0])
            inputs.append(row[1])
            struct_masks.append(row[2].le(self.struct_walk_size))
            struct_masks.append(row[3].le(self.struct_walk_size))
            labels.append(row[4])

        labels = torch.cat(labels, dim=0)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # inputs = torch.cat(inputs, dim=0)
        # False - is pad

        attention_masks = inputs.ne(self.tokenizer.pad_token_id)

        padded_struct_masks = []
        pad_length = inputs.size(1)
        for i in range(len(struct_masks)):
            struct_mask = struct_masks[i]

            padded_struct_mask = pad(struct_mask,
                                     [0, pad_length - struct_mask.size(1), 0, pad_length - struct_mask.size(0)],
                                     mode="constant", value=False)
            padded_struct_mask[struct_mask.size(0):, pad_length - 1] = True

            padded_struct_mask = padded_struct_mask.unsqueeze(0)
            padded_struct_masks.append(padded_struct_mask)

        struct_masks = torch.cat(padded_struct_masks, dim=0)
        return inputs, attention_masks, struct_masks, labels


class CloneDetectionBCBNoStructDataset(Dataset):
    def __init__(self, data_path, src_data_path, idx_file_path, tokenizer, max_seq_len=512, overwrite=False):
        self.data_path = data_path
        self.processed_data_names = []

        src_data = open(src_data_path)
        idx2code = {}
        for line in src_data:
            line = line.strip()
            line_data = json.loads(line)
            idx2code[line_data["idx"]] = line_data["func"]

        idxs = []
        idx_file = open(idx_file_path)
        i = 0
        for line in idx_file:
            line = line.strip()
            idx1, idx2, label = line.split('\t')
            label = int(label)
            if idx1 in idx2code and idx2 in idx2code:
                self.processed_data_names.append(f"processed_data_{str(i)}")
                idxs.append([idx1, idx2, label])
                i += 1

        src_data.close()
        idx_file.close()

        # process
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        print("start making dataset")

        with tqdm(total=len(self.processed_data_names)) as bar:
            for i, processed_data_name in enumerate(self.processed_data_names):
                processed_data_path = os.path.join(self.data_path, processed_data_name)
                if not os.path.exists(processed_data_path) or overwrite:
                    if not os.path.exists(processed_data_path):
                        os.makedirs(processed_data_path)
                    idx1, idx2, label = idxs[i]
                    code1 = idx2code[idx1]
                    code2 = idx2code[idx2]

                    code1 = ' '.join(code1.split())
                    code2 = ' '.join(code2.split())

                    code1_input_ids = tokenizer(code1, truncation=True, max_length=max_seq_len).input_ids
                    code2_input_ids = tokenizer(code2, truncation=True, max_length=max_seq_len).input_ids

                    input1 = torch.tensor(code1_input_ids, dtype=torch.long)
                    input2 = torch.tensor(code2_input_ids, dtype=torch.long)
                    label = torch.tensor([label], dtype=torch.long)

                    torch.save(input1, os.path.join(processed_data_path, "input1.pt"))
                    torch.save(input2, os.path.join(processed_data_path, "input2.pt"))
                    torch.save(label, os.path.join(processed_data_path, "label.pt"))

                bar.update(1)

    def __len__(self):
        return len(self.processed_data_names)

    def __getitem__(self, item):
        input1 = torch.load(os.path.join(self.data_path, self.processed_data_names[item], "input1.pt"))
        input2 = torch.load(os.path.join(self.data_path, self.processed_data_names[item], "input2.pt"))
        label = torch.load(os.path.join(self.data_path, self.processed_data_names[item], "label.pt"))
        return input1, input2, label


class CloneDetectionBCBNoStructCollater:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        inputs = []
        labels = []
        for row in data:
            inputs.append(row[0])
            inputs.append(row[1])
            labels.append(row[2])

        labels = torch.cat(labels, dim=0)
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # False - is pad
        attention_masks = padded_inputs.ne(self.tokenizer.pad_token_id)
        return padded_inputs, attention_masks, labels


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

                    # # self
                    # edge_index = max(cut_hyperedge_indexs[1]) + 1
                    # for i in range(len(tokens)):
                    #     cut_hyperedge_indexs[0].append(i)
                    #     cut_hyperedge_indexs[1].append(edge_index)
                    #     cut_edge_types.append(3)
                    #     edge_index += 1
                    # #

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
        input1 = torch.load(os.path.join(self.data_path, self.idx[item][0], "input.pt"))
        input2 = torch.load(os.path.join(self.data_path, self.idx[item][1], "input.pt"))
        hyperedge_indexs1 = torch.load(os.path.join(self.data_path, self.idx[item][0], "hyperedge_indexs.pt"))
        hyperedge_indexs2 = torch.load(os.path.join(self.data_path, self.idx[item][1], "hyperedge_indexs.pt"))
        edge_types1 = torch.load(os.path.join(self.data_path, self.idx[item][0], "edge_types.pt"))
        edge_types2 = torch.load(os.path.join(self.data_path, self.idx[item][1], "edge_types.pt"))

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
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_inputs = pad(padded_inputs, [0, 1, 0, 0], mode="constant", value=self.tokenizer.pad_token_id)

        # False - is pad
        attention_masks = padded_inputs.ne(self.tokenizer.pad_token_id)

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


class CloneDetectionBCBHyperAblationCollater:
    def __init__(self, tokenizer, ablation_type):
        self.tokenizer = tokenizer
        self.ablation_type = ablation_type

    def __call__(self, data):
        inputs = []
        hyperedge_indexs = []
        edge_types = []
        labels = []
        for row in data:
            inputs.append(row[0])
            inputs.append(row[1])

            hyperedge_index1 = row[2].transpose(0, 1)
            edge_type1 = row[4]
            mask1 = edge_type1 == self.ablation_type
            hyperedge_index1 = hyperedge_index1[mask1]
            hyperedge_indexs.append(hyperedge_index1.transpose(0, 1))
            edge_types.append(edge_type1[mask1])
            hyperedge_index2 = row[3].transpose(0, 1)
            edge_type2 = row[5]
            mask2 = edge_type2 == self.ablation_type
            hyperedge_index2 = hyperedge_index2[mask2]
            hyperedge_indexs.append(hyperedge_index2.transpose(0, 1))
            edge_types.append(edge_type2[mask2])

            labels.append(row[6])

        labels = torch.cat(labels, dim=0)
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_inputs = pad(padded_inputs, [0, 1, 0, 0], mode="constant", value=self.tokenizer.pad_token_id)

        # False - is pad
        attention_masks = padded_inputs.ne(self.tokenizer.pad_token_id)

        padded_hyperedge_indexs = []
        pad_length = max([hyperedge_index.size(1) for hyperedge_index in hyperedge_indexs])
        max_edge_plus1 = int(
            max(
                [hyperedge_index[1, :].max() if hyperedge_index.numel() != 0 else 0 for hyperedge_index in
                 hyperedge_indexs]
            )) + 1
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


class CloneDetectionBCBCodellamaDataset(Dataset):
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

                    tokens = [tokenizer.bos_token] + tokens[:max_seq_len - 2] + [tokenizer.eos_token]
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
        input1 = torch.load(os.path.join(self.data_path, self.idx[item][0], "input.pt"))
        input2 = torch.load(os.path.join(self.data_path, self.idx[item][1], "input.pt"))
        hyperedge_indexs1 = torch.load(os.path.join(self.data_path, self.idx[item][0], "hyperedge_indexs.pt"))
        hyperedge_indexs2 = torch.load(os.path.join(self.data_path, self.idx[item][1], "hyperedge_indexs.pt"))
        edge_types1 = torch.load(os.path.join(self.data_path, self.idx[item][0], "edge_types.pt"))
        edge_types2 = torch.load(os.path.join(self.data_path, self.idx[item][1], "edge_types.pt"))

        label = torch.tensor([self.idx[item][2]], dtype=torch.long)
        return input1, input2, hyperedge_indexs1, hyperedge_indexs2, edge_types1, edge_types2, label


class CloneDetectionBCBCodellamaCollater:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        inputs = []
        hyperedge_indexs = []
        edge_types = []
        labels = []
        inputs_length = []
        for row in data:
            inputs.append(row[0])
            inputs.append(row[1])
            hyperedge_indexs.append(row[2])
            hyperedge_indexs.append(row[3])
            edge_types.append(row[4])
            edge_types.append(row[5])
            labels.append(row[6])
            inputs_length.append(row[0].size(0))
            inputs_length.append(row[1].size(0))

        labels = torch.cat(labels, dim=0)
        inputs = self.tokenizer.pad({"input_ids": inputs})
        padded_inputs = inputs["input_ids"]
        padded_inputs = pad(padded_inputs, [1, 0, 0, 0], mode="constant", value=self.tokenizer.pad_token_id)

        # False - is pad
        attention_masks = inputs["attention_mask"]
        attention_masks = pad(attention_masks, [1, 0, 0, 0], mode="constant", value=0)

        padded_hyperedge_indexs = []
        pad_length = max([hyperedge_index.size(1) for hyperedge_index in hyperedge_indexs])
        max_edge_plus1 = int(max([hyperedge_index[1, :].max() for hyperedge_index in hyperedge_indexs])) + 1
        for i in range(len(hyperedge_indexs)):
            hyperedge_index = hyperedge_indexs[i]
            hyperedge_index[0] += padded_inputs.size(1) - inputs_length[i]

            padded_hyperedge_index = torch.zeros(2, pad_length, dtype=torch.long)
            padded_hyperedge_index[0] = pad(hyperedge_index[0], [0, pad_length - hyperedge_index.size(1)],
                                            mode="constant", value=0)
            padded_hyperedge_index[1] = pad(hyperedge_index[1], [0, pad_length - hyperedge_index.size(1)],
                                            mode="constant", value=max_edge_plus1)

            padded_hyperedge_index = padded_hyperedge_index.unsqueeze(0)
            padded_hyperedge_indexs.append(padded_hyperedge_index)
        padded_hyperedge_indexs = torch.cat(padded_hyperedge_indexs, dim=0)

        padded_edge_types = pad_sequence(edge_types, batch_first=True, padding_value=0)

        return padded_inputs, attention_masks, padded_hyperedge_indexs, padded_edge_types, labels
