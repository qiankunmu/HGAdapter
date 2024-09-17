import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import CodeLlamaTokenizer, GenerationConfig
import numpy as np
from tree_sitter import Language, Parser
import tree_sitter_ruby, tree_sitter_javascript, tree_sitter_java, tree_sitter_python, tree_sitter_php, tree_sitter_go
import evaluate
from bleu.bleu import Bleu

from MyDataset import CodeSummarizationCSNCodellamaDataset, CodeSummarizationCSNCodellamaCollater
import sys

sys.path.append("..")
from models.HyperStructAdapterConfig import HyperStructAdapterConfig
from models.LlamaHyperStructAdapterModels import MyLlamaHyperStructAdapterForCausalLM


def main():
    pretrain_model_name_or_path = "CodeLlama-7b"
    src_data_dir = "data/dataset"
    data_dir = "data/processed_data_codellama"
    output_dir = "work_dir/codellama-hyperstruct-adapter"

    train_batch_size = 64
    eval_batch_size = 1
    learning_rate = 1e-4
    num_epochs = 3
    print_steps = 1000
    max_seq_len = 512
    max_tgt_len = 128
    reduction_factor = 64
    num_heads = 8
    dropout_out = 0.2
    use_norm = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patience = 2
    model_name = "codellama-hyperstruct-adapter"
    parameters = f"batch size={train_batch_size} learning rate={learning_rate} num_heads={num_heads} dropout={dropout_out}"
    mini_batch_size = 1
    assert train_batch_size % mini_batch_size == 0, "train_batch_size can not be divisible by mini_batch_size"
    num_mini_batch = train_batch_size // mini_batch_size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_save_path = os.path.join(output_dir, "results.csv")
    model_save_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    tokenizer = CodeLlamaTokenizer.from_pretrained(pretrain_model_name_or_path)
    tokenizer.pad_token_id = tokenizer.unk_token_id

    language_dict = {"ruby": tree_sitter_ruby.language(), "python": tree_sitter_python.language(),
                     "java": tree_sitter_java.language(),
                     "javascript": tree_sitter_javascript.language(), "go": tree_sitter_go.language(),
                     "php": tree_sitter_php.language_php()}

    languages = ["ruby", "python", "java", "javascript", "go", "php"]
    # languages = ["ruby"]
    result_record = {}
    for language in languages:
        print(f"start {language}")

        data_path = os.path.join(data_dir, language)
        train_data_path = os.path.join(data_path, "train")
        valid_data_path = os.path.join(data_path, "valid")
        test_data_path = os.path.join(data_path, "test")

        src_data_path = os.path.join(src_data_dir, language)
        train_src_data_path = os.path.join(src_data_path, "train.jsonl")
        valid_src_data_path = os.path.join(src_data_path, "valid.jsonl")
        test_src_data_path = os.path.join(src_data_path, "test.jsonl")

        model_save_path = os.path.join(model_save_dir, f"{language}")
        text_save_path = os.path.join(model_save_dir, f"{language}-pred-texts.txt")
        language_result_save_path = os.path.join(output_dir, f"{language}-results.csv")

        LANGUAGE = Language(language_dict[language])
        parser = Parser(LANGUAGE)

        collate_fn = CodeSummarizationCSNCodellamaCollater(tokenizer)
        collate_fn2 = CodeSummarizationCSNCodellamaCollater(tokenizer, "valid")

        train_dataset = CodeSummarizationCSNCodellamaDataset(train_data_path, train_src_data_path, tokenizer, parser,
                                                             max_seq_len=max_seq_len, max_tgt_len=max_tgt_len,
                                                             overwrite=False)
        train_sampler = RandomSampler(train_dataset, num_samples=len(train_dataset))
        train_dataloader = DataLoader(train_dataset, mini_batch_size, sampler=train_sampler, collate_fn=collate_fn)
        valid_dataset = CodeSummarizationCSNCodellamaDataset(valid_data_path, valid_src_data_path, tokenizer, parser,
                                                             max_seq_len=max_seq_len, max_tgt_len=max_tgt_len,
                                                             overwrite=False, mode="valid")
        valid_dataloader = DataLoader(valid_dataset, eval_batch_size, False, collate_fn=collate_fn2)
        test_dataset = CodeSummarizationCSNCodellamaDataset(test_data_path, test_src_data_path, tokenizer, parser,
                                                            max_seq_len=max_seq_len, max_tgt_len=max_tgt_len,
                                                            overwrite=False, mode="test")
        test_dataloader = DataLoader(test_dataset, eval_batch_size, False, collate_fn=collate_fn2)

        # Initialize model
        model = MyLlamaHyperStructAdapterForCausalLM.from_pretrained(pretrain_model_name_or_path,
                                                                     torch_dtype=torch.bfloat16,
                                                                     pad_token_id=tokenizer.pad_token_id)
        adapter_config = HyperStructAdapterConfig(use_hyper=True, reduction_factor=reduction_factor,
                                                  num_heads=num_heads, dropout=dropout_out, use_norm=use_norm,
                                                  torch_dtype="bfloat16")
        model.model.add_adapter("code-summarization", config=adapter_config)
        model.model.set_active_adapters("code-summarization")
        model.model.train_adapter("code-summarization")
        model.to(torch.bfloat16)

        model.to(device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=patience - 1,
                                                               threshold=1e-6, verbose=True, eps=1e-12)

        best_score = 0
        best_bleu4_micro = 0
        best_epoch = 0
        attk = 0
        for epoch in tqdm(range(num_epochs)):
            model.train()
            train_loss = 0
            with tqdm(total=len(train_dataloader)) as bar:
                for i, (input_ids, attention_mask, hyperedge_indexs, edge_types, labels) in enumerate(train_dataloader):
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    attention_mask = attention_mask.to(device)
                    hyperedge_indexs = hyperedge_indexs.to(device)
                    edge_types = edge_types.to(device)
                    output = model(input_ids, attention_mask, labels=labels, hyperedge_indexs=hyperedge_indexs,
                                   edge_types=edge_types)

                    loss = output.loss
                    train_loss += loss.item()

                    loss = loss / num_mini_batch
                    loss.backward()

                    if (i + 1) % num_mini_batch == 0 or (i + 1) == len(train_dataloader):
                        optimizer.step()
                        optimizer.zero_grad()

                    if i % (print_steps) == 0:
                        print(f"{language} epoch={epoch} loss={train_loss / (i + 1)}")
                    bar.update(1)

            optimizer.zero_grad()

            train_loss /= len(train_dataloader)
            print(f"{language} epoch {epoch} finish train loss={train_loss}")
            valid_bleu4_micro, valid_bleu4_macro = eval(model, valid_dataloader, tokenizer, device, text_save_path)
            print(f"valid bleu4-macro={valid_bleu4_macro} bleu4-micro={valid_bleu4_micro}")

            scheduler.step(valid_bleu4_macro)

            if valid_bleu4_macro > best_score:
                best_score = valid_bleu4_macro
                best_epoch = epoch
                best_bleu4_micro = valid_bleu4_micro
                model.model.save_adapter(model_save_path, "code-summarization")
                save_state = {state: model.state_dict()[state] for state in model.state_dict() if "lm_head" in state}
                torch.save(save_state, os.path.join(model_save_path, "lm_head.bin"))
                print("successfully save")
                attk = 0
            else:
                print("no better than last best time")
                attk += 1
                if attk >= patience:
                    attk = 0
                    print("reload last saved model")
                    model.model.load_adapter(model_save_path)
                    model.load_state_dict(torch.load(os.path.join(model_save_path, "lm_head.bin")), strict=False)
                    model.to(torch.bfloat16)
                    model.to(device)

        print(f"finish {language} training")
        print(f"best valid bleu4-macro={best_score} bleu4-micro={best_bleu4_micro} epoch={best_epoch}")

        model.model.load_adapter(model_save_path)
        model.load_state_dict(torch.load(os.path.join(model_save_path, "lm_head.bin")), strict=False)
        model.to(torch.bfloat16)
        model.to(device)

        test_bleu4_micro, test_bleu4_macro = eval(model, test_dataloader, tokenizer, device, text_save_path)
        print(f"{language} test bleu4-macro={test_bleu4_macro} bleu4-micro={test_bleu4_micro}")

        result_record[f"{language} best valid bleu4-macro"] = best_score
        result_record[f"{language} valid bleu4-micro"] = best_bleu4_micro
        result_record[f"{language} best epoch"] = best_epoch
        result_record[f"{language} test bleu4-macro"] = test_bleu4_macro
        result_record[f"{language} test bleu4-micro"] = test_bleu4_micro

        title = ["model name", "best valid bleu4-macro", "valid bleu4-micro", "test bleu4-macro", "test bleu4-micro",
                 "best epoch", "parameters"]
        if not os.path.exists(language_result_save_path):
            df = pd.DataFrame(
                [[model_name, best_score, best_bleu4_micro, test_bleu4_macro, test_bleu4_micro, best_epoch,
                  parameters]],
                columns=title)
            df.to_csv(language_result_save_path, index=False, header=title)
        else:
            df = pd.read_csv(language_result_save_path)
            df.loc[len(df.index)] = [model_name, best_score, best_bleu4_micro, test_bleu4_macro, test_bleu4_micro,
                                     best_epoch, parameters]
            df.to_csv(language_result_save_path, index=False, header=title)

    if len(languages) == 6:
        title = ["model name"]
        result = [model_name]
        for language in languages:
            title.extend(
                [f"{language} best valid bleu4-macro", f"{language} valid bleu4-micro", f"{language} test bleu4-macro",
                 f"{language} test bleu4-micro", f"{language} best epoch"])
            result.extend(
                [result_record[f"{language} best valid bleu4-macro"], result_record[f"{language} valid bleu4-micro"],
                 result_record[f"{language} test bleu-macro"], result_record[f"{language} test bleu-micro"],
                 result_record[f"{language} best epoch"]])
        title.append("parameters")
        result.append(parameters)

        if not os.path.exists(result_save_path):
            df = pd.DataFrame([result], columns=title)
            df.to_csv(result_save_path, index=False, header=title)
        else:
            df = pd.read_csv(result_save_path)
            df.loc[len(df.index)] = result
            df.to_csv(result_save_path, index=False, header=title)

    print("all finish")


def eval(model, dataloader, tokenizer, device, text_save_path=None):
    global file, file2
    print("start evaluate...")

    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.max_new_tokens = 128
    generation_config.num_beams = 4

    model.eval()
    preds_texts, labels_texts = [], []
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as bar:
            for input_ids, attention_mask, hyperedge_indexs, edge_types, labels in dataloader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                hyperedge_indexs = hyperedge_indexs.to(device)
                edge_types = edge_types.to(device)
                generated_ids = model.generate(input_ids, generation_config, hyperedge_indexs=hyperedge_indexs,
                                               edge_types=edge_types)
                preds = generated_ids[:, input_ids.shape[1]:]
                preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
                labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

                for pred_text, label_text in zip(preds_text, labels_text):
                    preds_texts.append(pred_text.strip().lower())
                    labels_texts.append([label_text.strip().lower()])

                bar.update(1)

    if text_save_path:
        file = open(text_save_path, "w+")
        file2 = open(text_save_path[:-14] + "refer-texts.txt", "w+")

    bleu = Bleu()
    bleu4_micro = bleu.compute(predictions=preds_texts, references=labels_texts, smooth=True)["bleu"]
    bleu4_macros = []
    for pred_text, label_text in zip(preds_texts, labels_texts):
        if len(pred_text) == 0:
            pred_text = "<pad>"
        bleu4_macros.append(bleu.compute(predictions=[pred_text], references=[label_text], smooth=True)["bleu"])
        if text_save_path:
            file.write(pred_text + "\n")
            file2.write(label_text[0] + "\n")

    bleu4_macro = np.mean(bleu4_macros)
    if text_save_path:
        file.close()
        file2.close()

    return bleu4_micro, bleu4_macro


if __name__ == "__main__":
    main()
