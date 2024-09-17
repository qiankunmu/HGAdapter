import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizer, EncoderDecoderConfig, RobertaForCausalLM, \
    GenerationConfig
import numpy as np
from tree_sitter import Language, Parser
import tree_sitter_ruby, tree_sitter_javascript, tree_sitter_java, tree_sitter_python, tree_sitter_go, tree_sitter_php
import evaluate
from bleu.bleu import Bleu

from MyDataset import CodeSummarizationCSNHyperDataset, CodeSummarizationCSNHyperCollater
from model import MyHyperEncoderDecoderModel
import sys

sys.path.append("..")
from models.RobertaHyperStructAdapterModels import MyRobertaHyperStructAdapterModel
from models.HyperStructAdapterConfig import HyperStructAdapterConfig


def main():
    pretrain_model_name_or_path = "codebert-base"
    src_data_dir = "data/dataset"
    data_dir = "data/processed_data_hyper"
    output_dir = "work_dir/codebert-hyperstruct-adapter"

    train_batch_size = 64
    eval_batch_size = 4
    learning_rate = 1e-4
    num_epochs = 10
    print_steps = 1000
    max_seq_len = 512
    max_tgt_len = 128
    reduction_factor = 12
    num_heads = 8
    dropout_out = 0.2
    use_norm = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    patience = 2
    model_name = "codebert-hyperstruct-adapter"
    parameters = f"batch size={train_batch_size} learning rate={learning_rate} reduction_factor={reduction_factor} num_heads={num_heads} dropout={dropout_out}"
    mini_batch_size = 4
    assert train_batch_size % mini_batch_size == 0, "train_batch_size can not be divisible by mini_batch_size"
    num_mini_batch = train_batch_size // mini_batch_size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_save_path = os.path.join(output_dir, "results.csv")
    model_save_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_name_or_path)

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

        # Initialize model
        encoder = MyRobertaHyperStructAdapterModel.from_pretrained(pretrain_model_name_or_path)

        decoder_config = RobertaConfig(vocab_size=encoder.config.vocab_size, bos_token_id=encoder.config.bos_token_id,
                                       eos_token_id=encoder.config.eos_token_id,
                                       pad_token_id=encoder.config.pad_token_id, hidden_size=encoder.config.hidden_size,
                                       type_vocab_size=encoder.config.type_vocab_size,
                                       num_hidden_layers=12, max_position_embeddings=max_tgt_len + 2, is_decoder=True,
                                       add_cross_attention=True)
        decoder = RobertaForCausalLM(decoder_config)
        # decoder.from_pretrained(pretrain_model_name_or_path)

        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config,
                                                                   decoder_start_token_id=tokenizer.cls_token_id,
                                                                   pad_token_id=tokenizer.pad_token_id)
        model = MyHyperEncoderDecoderModel(config, encoder=encoder, decoder=decoder)

        adapter_config = HyperStructAdapterConfig(use_hyper=True, reduction_factor=reduction_factor,
                                                  num_heads=num_heads, dropout=dropout_out, use_norm=use_norm)
        encoder.add_adapter("code-summarization", config=adapter_config)
        encoder.set_active_adapters("code-summarization")
        encoder.train_adapter("code-summarization")

        model.to(device)

        collate_fn = CodeSummarizationCSNHyperCollater(tokenizer)

        train_dataset = CodeSummarizationCSNHyperDataset(train_data_path, train_src_data_path, tokenizer, parser,
                                                         max_seq_len=max_seq_len, max_tgt_len=max_tgt_len,
                                                         overwrite=False)
        train_dataloader = DataLoader(train_dataset, mini_batch_size, True, collate_fn=collate_fn)
        valid_dataset = CodeSummarizationCSNHyperDataset(valid_data_path, valid_src_data_path, tokenizer, parser,
                                                         max_seq_len=max_seq_len, max_tgt_len=max_tgt_len,
                                                         overwrite=False)
        valid_dataloader = DataLoader(valid_dataset, eval_batch_size, False, collate_fn=collate_fn)
        test_dataset = CodeSummarizationCSNHyperDataset(test_data_path, test_src_data_path, tokenizer, parser,
                                                        max_seq_len=max_seq_len, max_tgt_len=max_tgt_len,
                                                        overwrite=False)
        test_dataloader = DataLoader(test_dataset, eval_batch_size, False, collate_fn=collate_fn)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        num_steps = (((len(train_dataloader) - 1) // num_mini_batch) + 1) * num_epochs
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
                for i, (input_ids, labels, attention_mask, hyperedge_indexs, edge_types) in enumerate(train_dataloader):
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

                    if i % print_steps == 0:
                        print(f"{language} epoch={epoch} loss={train_loss / (i + 1)}")
                    bar.update(1)

            optimizer.zero_grad()

            train_loss /= len(train_dataloader)
            print(f"{language} epoch {epoch} finish train loss={train_loss}")
            valid_loss, valid_bleu4_micro, valid_bleu4_macro = eval(model, valid_dataloader, tokenizer, device,
                                                                    text_save_path)
            print(f"valid loss={valid_loss} bleu4-macro={valid_bleu4_macro} bleu4-micro={valid_bleu4_micro}")

            scheduler.step(valid_bleu4_macro)

            if valid_bleu4_macro > best_score:
                best_score = valid_bleu4_macro
                best_epoch = epoch
                best_bleu4_micro = valid_bleu4_micro
                model.encoder.base_model.save_adapter(model_save_path, "code-summarization")
                torch.save(model.decoder.state_dict(), os.path.join(model_save_path, "decoder_state.bin"))
                print("successfully save")
                attk = 0
            else:
                attk += 1
                print("no better than last best time")
                if attk >= patience:
                    attk = 0
                    print("reload last saved model")
                    model.encoder.base_model.load_adapter(model_save_path)
                    model.decoder.load_state_dict(torch.load(os.path.join(model_save_path, "decoder_state.bin")))

        print(f"finish {language} training")
        print(f"best valid bleu4-macro={best_score} bleu4-micro={best_bleu4_micro} epoch={best_epoch}")

        model.encoder.base_model.load_adapter(model_save_path)
        model.decoder.load_state_dict(torch.load(os.path.join(model_save_path, "decoder_state.bin")))
        model.to(device)

        test_loss, test_bleu4_micro, test_bleu4_macro = eval(model, test_dataloader, tokenizer, device, text_save_path)
        print(f"{language} test loss={test_loss} bleu4-macro={test_bleu4_macro} bleu4-micro={test_bleu4_micro}")

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
    global file2, file
    print("start evaluate...")

    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.max_new_tokens = 128
    generation_config.num_beams = 4

    model.eval()
    losses = []
    preds_texts, labels_texts = [], []
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as bar:
            for input_ids, labels, attention_mask, hyperedge_indexs, edge_types in dataloader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)
                hyperedge_indexs = hyperedge_indexs.to(device)
                edge_types = edge_types.to(device)
                loss = model(input_ids, attention_mask, labels=labels, hyperedge_indexs=hyperedge_indexs,
                             edge_types=edge_types).loss
                losses.append(loss.item())
                preds = model.generate(input_ids, generation_config, attention_mask=attention_mask,
                                       hyperedge_indexs=hyperedge_indexs, edge_types=edge_types)
                labels.masked_fill_(labels == -100, tokenizer.pad_token_id)
                preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
                labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

                for pred_text, label_text in zip(preds_text, labels_text):
                    preds_texts.append(pred_text.strip().lower())
                    labels_texts.append([label_text.strip().lower()])

                bar.update(1)

    if text_save_path:
        file = open(text_save_path, "w+")
        file2 = open(text_save_path[:-14] + "refer-texts.txt", "w+")

    # you can also use this bleu
    # bleu = evaluate.load("bleu.py")
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
    avg_loss = np.mean(losses)
    if text_save_path:
        file.close()
        file2.close()

    return avg_loss, bleu4_micro, bleu4_macro


def run_test():
    pretrain_model_name_or_path = "../pre_train_models/codebert-base"
    src_data_dir = "data/dataset"
    data_dir = "data/processed_data_hyper"
    output_dir = "work_dir/codebert-hyperstruct-adapter"

    train_batch_size = 64
    eval_batch_size = 32
    learning_rate = 5e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "codebert-hyperstruct-adapter1"
    parameters = f"batch size={train_batch_size} learning rate={learning_rate} adam"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_save_path = os.path.join(output_dir, "results.csv")
    model_save_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_name_or_path)

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
        test_data_path = os.path.join(data_path, "test")

        src_data_path = os.path.join(src_data_dir, language)
        test_src_data_path = os.path.join(src_data_path, "test.jsonl")

        model_save_path = os.path.join(model_save_dir, f"{language}")
        text_save_path = os.path.join(model_save_dir, f"{language}-pred-texts.txt")
        language_result_save_path = os.path.join(output_dir, f"{language}-results.csv")

        LANGUAGE = Language(language_dict[language])
        parser = Parser(LANGUAGE)

        # Initialize model
        encoder = MyRobertaHyperStructAdapterModel.from_pretrained(pretrain_model_name_or_path)

        decoder_config = RobertaConfig(vocab_size=encoder.config.vocab_size, bos_token_id=encoder.config.bos_token_id,
                                       eos_token_id=encoder.config.eos_token_id,
                                       pad_token_id=encoder.config.pad_token_id, hidden_size=encoder.config.hidden_size,
                                       type_vocab_size=encoder.config.type_vocab_size,
                                       num_hidden_layers=encoder.config.num_hidden_layers, is_decoder=True,
                                       add_cross_attention=True)
        decoder = RobertaForCausalLM(decoder_config)

        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder_config,
                                                                   decoder_start_token_id=tokenizer.cls_token_id,
                                                                   pad_token_id=tokenizer.pad_token_id)
        model = MyHyperEncoderDecoderModel(config, encoder=encoder, decoder=decoder)

        adapter_config = HyperStructAdapterConfig(use_hyper=True)
        encoder.add_adapter("code-summarization", config=adapter_config)
        encoder.set_active_adapters("code-summarization")
        encoder.train_adapter("code-summarization")

        model.encoder.base_model.load_adapter(model_save_path)
        model.decoder.load_state_dict(torch.load(os.path.join(model_save_path, "decoder_state.bin")))
        model.to(device)

        collate_fn = CodeSummarizationCSNHyperCollater(tokenizer)

        test_dataset = CodeSummarizationCSNHyperDataset(test_data_path, test_src_data_path, tokenizer, parser,
                                                        max_seq_len=512, overwrite=False)
        test_dataloader = DataLoader(test_dataset, eval_batch_size, False, collate_fn=collate_fn)

        test_loss, test_bleu4_micro, test_bleu4_macro = eval(model, test_dataloader, tokenizer, device, text_save_path)
        print(f"{language} test loss={test_loss} bleu4-macro={test_bleu4_macro} bleu4-micro={test_bleu4_micro}")


if __name__ == "__main__":
    main()
