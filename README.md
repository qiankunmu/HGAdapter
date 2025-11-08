# Hypergraph-based Adapter (HGAdapter)
## Introduction
This is the source code implementation of Hypergraph-based Adapter (HGAdapter) in paper [*HGAdapter: Hypergraph-based Adapters in Language Models for Code Summarization and Clone Detection*](https://aclanthology.org/2025.findings-emnlp.800/). 
Our paper is accepted by the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025) as a findings long paper.
We will update related content soon.

## Related Links
[![paper](https://img.shields.io/badge/EMNLP_2025-Findings-gray?labelColor=E60012)](https://aclanthology.org/2025.findings-emnlp.800/) &nbsp;&nbsp; Official paper link [https://aclanthology.org/2025.findings-emnlp.800/](https://aclanthology.org/2025.findings-emnlp.800/)  
[![arxiv](https://img.shields.io/badge/arXiv-2510.17591-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.17591) &nbsp;&nbsp; Arxiv preprint paper link [https://arxiv.org/abs/2510.17591](https://arxiv.org/abs/2510.17591)  
[![GitHub](https://img.shields.io/badge/GitHub-HGAdapter-181717?logo=github&logoColor=white)](https://github.com/qiankunmu/HGAdapter) &nbsp; Code implementation link [https://github.com/qiankunmu/HGAdapter](https://github.com/qiankunmu/HGAdapter)

## Requirments
We conduct experiment in Ubuntu 18.04.6 LTS and in python 3.12. 
We mainly implement our method by [pytorch 2.4](https://pytorch.org/docs/stable/index.html), [transformers 4.48](https://huggingface.co/docs/transformers) and [adapters 1.1](https://docs.adapterhub.ml/). 
We train our model on a RTX 3090. 
The required environments are listed in `requirements.txt`.

## Datasets
### CodeSearchNet
CodeSearchNet is the dataset for code summarization task. 
We use the edition from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE). 
The following process is adapted from CodeXGLUE, you can also refer to [this](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text).  

First, you need to download [datasets.zip](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Text/code-to-text/dataset.zip), then put it into `Code-summarization-CodeSearchNet/data`.  
Then
```
unzip dataset.zip
cd dataset
```  

Second, you can download [ruby.zip](https://huggingface.co/datasets/code-search-net/code_search_net/blob/main/data/ruby.zip), [javascript.zip](https://huggingface.co/datasets/code-search-net/code_search_net/blob/main/data/javascript.zip), [java.zip](https://huggingface.co/datasets/code-search-net/code_search_net/blob/main/data/java.zip), [python.zip](https://huggingface.co/datasets/code-search-net/code_search_net/blob/main/data/python.zip), [php.zip](https://huggingface.co/datasets/code-search-net/code_search_net/blob/main/data/php.zip) and [go.zip](https://huggingface.co/datasets/code-search-net/code_search_net/blob/main/data/go.zip) from [Hugging Face](https://huggingface.co/datasets/code-search-net/code_search_net/tree/main/data).  
Then, put them into `Code-summarization-CodeSearchNet/data/dataset` and unzip them. 
```
unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl
```  

Third, run `preprocess.py` to get the data. 
```
python preprocess.py
rm -r */final
rm -r */*.txt
```

### BigCloneBench
BigCloneBench is the dataset for code clone detection task. 
We use the edition from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE). 
You need to download the [datasets](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench/dataset). 
You can download [data.jsonl](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Clone-detection-BigCloneBench/dataset/data.jsonl), [train.txt](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Clone-detection-BigCloneBench/dataset/train.txt), [valid.txt](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Clone-detection-BigCloneBench/dataset/valid.txt) and [test.txt](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Clone-detection-BigCloneBench/dataset/test.txt), then put them into `Clone-detection-BigCloneBench/data/dataset`. 


## Usages
1\. For code summarization task, you can run `trainLlama.py`, it can train, valid and test the Llama series models with HGAdapter inserted. 
You can modify `pretrain_model_name_or_path` in `trainLlama.py` to choose the pre-trained model, such as 
```
pretrain_model_name_or_path = "CodeLlama-7b"
```
You can also choose `"TinyLlama_v1.1_math_code"` or other Llama series models.  

You can run `trainHyper.py`, it can train, valid and test the RoBERTa series models with HGAdapter inserted. 
You can modify `pretrain_model_name_or_path` in `trainHyper.py` to choose the pre-trained model, such as 
```
pretrain_model_name_or_path = "codebert-base"
```
You can also choose `"roberta-base"`, `"graphcodebert-base"`, `"unixcoder-base"` or other RoBERTa series models.  

You can run `trainQwen2.py`, it can train, valid and test the Qwen2.5 series models with HGAdapter inserted. 
You can modify `pretrain_model_name_or_path` in `trainQwen2.py` to choose the pre-trained model, such as 
```
pretrain_model_name_or_path = "Qwen2.5-Coder-0.5B"
```
You can also choose other Qwen2.5 series models.

2\. For code clone detection, you can run `trainHyper.py`, it can train, valid and test the RoBERTa series models with HGAdapter inserted. 
You can modify `pretrain_model_name_or_path` in `trainHyper.py` to choose the pre-trained model, such as 
```
pretrain_model_name_or_path = "codebert-base"
```
You can also choose `"graphcodebert-base"` and `"unixcoder-base"`. 

3\. You can also locally download pre-trained models into `pre_train_models/`, modify the `pretrain_model_name_or_path` to the corresponding path.  

4\. After the run is complete, the model will be saved to the `work_dir`, and the results will be outputted and saved as a CSV in `work_dir`.  
The implementation of the HGAdapter is located in the `models` folder.  
You can manually modify hyper-parameters such as `train_batch_size`, `num_epochs` in trainXXX.py.  

## Citation
If you use our code, please cite us.
```
@inproceedings{Yang2025HGAdapter,
  title={HGAdapter: Hypergraph-based Adapters in Language Models for Code Summarization and Clone Detection},
  author={Guang Yang and Yujie Zhu},
  booktitle={Findings of the Association for Computational Linguistics: {EMNLP} 2025},
  pages={14823–14833},
  year={2025}
}
```
