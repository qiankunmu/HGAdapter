import os

import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaConfig
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from tree_sitter import Language, Parser
import tree_sitter_java

from MyDataset import CloneDetectionBCBHyperDataset, CloneDetectionBCBHyperCollater
from model import CloneDetectionBCBHyperModel
import sys

sys.path.append("..")
from models.RobertaHGAdapterModels import MyRobertaHGAdapterModel
from models.HGAdapterConfig import HGAdapterConfig


def main():
    pretrain_model_name_or_path = "codebert-base"
    src_data_path = "data/dataset/data.jsonl"
    data_path = "data/processed_data_hyper"
    train_idx_file_path = "data/dataset/train.txt"
    valid_idx_file_path = "data/dataset/valid.txt"
    test_idx_file_path = "data/dataset/test.txt"
    output_dir = "work_dir/codebert-hgadapter"

    train_batch_size = 64
    eval_batch_size = 64
    learning_rate = 5e-5
    num_epochs = 1
    print_steps = 1000
    train_ratio = 0.1
    patience = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "codebert-hgadapter"
    parameters = f"batch size={train_batch_size} learning rate={learning_rate}"
    mini_batch_size = 64
    assert train_batch_size % mini_batch_size == 0, "train_batch_size can not be divisible by mini_batch_size"
    num_mini_batch = train_batch_size // mini_batch_size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_save_path = os.path.join(output_dir, model_name)
    result_save_path = os.path.join(output_dir, "results.csv")

    config = RobertaConfig.from_pretrained(pretrain_model_name_or_path)
    roberta_model = MyRobertaHGAdapterModel.from_pretrained(pretrain_model_name_or_path, architectures=["RobertaModel"])
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_name_or_path)

    LANGUAGE = Language(tree_sitter_java.language())
    parser = Parser(LANGUAGE)

    adapter_config = HGAdapterConfig(use_hyper=True, num_edge_types=3)
    roberta_model.add_adapter("clone-detection", config=adapter_config)
    roberta_model.set_active_adapters("clone-detection")
    roberta_model.train_adapter("clone-detection")
    model = CloneDetectionBCBHyperModel(roberta_model, config)

    model.to(device)

    collate_fn = CloneDetectionBCBHyperCollater(tokenizer)

    train_dataset = CloneDetectionBCBHyperDataset(data_path, src_data_path, train_idx_file_path, tokenizer, parser,
                                                  max_seq_len=512, overwrite=False)
    train_sampler = RandomSampler(train_dataset, num_samples=int(len(train_dataset) * train_ratio))
    train_dataloader = DataLoader(train_dataset, mini_batch_size, sampler=train_sampler, collate_fn=collate_fn)
    valid_dataset = CloneDetectionBCBHyperDataset(data_path, src_data_path, valid_idx_file_path, tokenizer, parser,
                                                  max_seq_len=512)
    valid_dataloader = DataLoader(valid_dataset, eval_batch_size, False, collate_fn=collate_fn)
    test_dataset = CloneDetectionBCBHyperDataset(data_path, src_data_path, test_idx_file_path, tokenizer, parser,
                                                 max_seq_len=512)
    test_dataloader = DataLoader(test_dataset, eval_batch_size, False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=patience - 1,
                                                           verbose=True, eps=1e-12)

    loss_function = torch.nn.CrossEntropyLoss()

    best_score = -1
    best_thresold, best_precision, best_recall, best_f1 = 0, 0, 0, 0
    best_epoch = 0
    attk = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        i = 0
        with tqdm(total=len(train_dataloader)) as bar:
            for i, (input_ids, attention_mask, hyperedge_indexs, edge_types, label) in enumerate(train_dataloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                hyperedge_indexs = hyperedge_indexs.to(device)
                edge_types = edge_types.to(device)
                label = label.to(device)
                output = model(input_ids, attention_mask, hyperedge_indexs, edge_types)
                loss = loss_function(output, label)
                train_loss += loss.item()

                loss = loss / num_mini_batch
                loss.backward()
                if (i + 1) % num_mini_batch == 0 or (i + 1) == len(train_dataloader):
                    optimizer.step()
                    optimizer.zero_grad()

                if i % print_steps == 0:
                    print(f"epoch={epoch} loss={train_loss / (i + 1)}")
                bar.update(1)

        optimizer.zero_grad()

        train_loss /= (i + 1)
        print(f"epoch {epoch} finish train loss={train_loss}")
        valid_loss, valid_thresold, valid_f1, valid_precision, valid_recall = evaluate(model, valid_dataloader, device)
        print(
            f"valid loss={valid_loss} best thresold={valid_thresold} best f1={valid_f1} precision={valid_precision} recall={valid_recall}")

        scheduler.step(valid_f1)

        if valid_f1 > best_score:
            best_score = valid_f1
            best_precision = valid_precision
            best_recall = valid_recall
            best_f1 = valid_f1
            best_epoch = epoch
            best_thresold = valid_thresold
            model.model.save_adapter(model_save_path, "clone-detection")
            save_state = {}
            for state in model.state_dict():
                if ("feed_forward" in state) or ("output_layer" in state):
                    save_state[state] = model.state_dict()[state]
            torch.save(save_state, os.path.join(model_save_path, "other_state.bin"))
            print("successfully save")
            attk = 0
        else:
            print("no better than last best time")
            attk += 1
            if attk >= patience:
                print("reload last best model")
                model.model.load_adapter(model_save_path)
                model.load_state_dict(torch.load(os.path.join(model_save_path, "other_state.bin")), strict=False)
                attk = 0

    print("finish training")
    print(
        f"best valid thresold={best_thresold} f1 score={best_f1} precision score={best_precision} recall score={best_recall} epoch={best_epoch}")

    model.model.load_adapter(model_save_path)
    model.load_state_dict(torch.load(os.path.join(model_save_path, "other_state.bin")), strict=False)
    model.to(device)

    test_loss, test_f1, test_precision, test_recall = tes(model, test_dataloader, best_thresold, device)
    print(f"test loss={test_loss} f1={test_f1} precision={test_precision} recall={test_recall}")

    title = ['model name', 'best valid f1', 'valid precision', 'valid recall', 'test f1', 'test precision',
             'test recall', 'best thresold', 'best epoch', 'parameters']
    if not os.path.exists(result_save_path):
        df = pd.DataFrame(
            [[model_name, best_f1, best_precision, best_recall, test_f1, test_precision, test_recall, best_thresold,
              best_epoch, parameters]], columns=title)

        df.to_csv(result_save_path, index=False, header=title)
    else:
        df = pd.read_csv(result_save_path)
        df.loc[len(df.index)] = [model_name, best_f1, best_precision, best_recall, test_f1, test_precision,
                                 test_recall, best_thresold, best_epoch, parameters]
        df.to_csv(result_save_path, index=False, header=title)

    print("finish")


def evaluate(model, dataloader, device):
    print("strat evaluate...")

    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    losses = []
    logits, labels = [], []
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as bar:
            for input_ids, attention_mask, hyperedge_indexs, edge_types, label in dataloader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                hyperedge_indexs = hyperedge_indexs.to(device)
                edge_types = edge_types.to(device)
                label = label.to(device)
                output = model(input_ids, attention_mask, hyperedge_indexs, edge_types)
                logit = torch.nn.functional.softmax(output, dim=-1)
                loss = loss_function(output, label)
                losses.append(loss.item())
                logits.extend(logit[0:, 1].cpu().detach().numpy().tolist())
                labels.extend(label.cpu().detach().numpy().tolist())

                bar.update(1)

    avg_loss = np.mean(losses)

    best_f1, best_precision, best_recall = 0, 0, 0
    best_thresold = 0
    for thresold in np.arange(0.01, 1, 0.1):
        preds = []
        for l in logits:
            if l > thresold:
                preds.append(1)
            else:
                preds.append(0)

        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_thresold = thresold

    return avg_loss, best_thresold, best_f1, best_precision, best_recall


def tes(model, dataloader, thresold, device):
    print("start test...")

    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    losses = []
    logits, labels = [], []
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as bar:
            for input_ids, attention_mask, hyperedge_indexs, edge_types, label in dataloader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                hyperedge_indexs = hyperedge_indexs.to(device)
                edge_types = edge_types.to(device)
                label = label.to(device)
                output = model(input_ids, attention_mask, hyperedge_indexs, edge_types)
                logit = torch.nn.functional.softmax(output, dim=-1)
                loss = loss_function(output, label)
                losses.append(loss.item())
                logits.extend(logit[0:, 1].cpu().detach().numpy().tolist())
                labels.extend(label.cpu().detach().numpy().tolist())

                bar.update(1)

    avg_loss = np.mean(losses)

    preds = []
    for l in logits:
        if l > thresold:
            preds.append(1)
        else:
            preds.append(0)

    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    return avg_loss, f1, precision, recall


if __name__ == "__main__":
    main()
