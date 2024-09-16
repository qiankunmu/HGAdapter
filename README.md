This the source code implementation of HGAdapter: Hypergraph-based Structural Adapter Fine-tuning Pre-Trained Model.  
Due to the paper being under peer review, in accordance with the principle of anonymity, we temporarily withhold the title, authors and contents of the paper.
## Requirments
We conduct experiment in Ubuntu 18.04.6 LTS and in python 3.12. 
We mainly implement our method by [pytorch 2.4](https://pytorch.org/docs/stable/index.html), [transformers 4.44](https://huggingface.co/docs/transformers) and [adapters 1.0](https://docs.adapterhub.ml/). 
We train our model on a RTX 3090. 
The required environments are listed in `requirements.txt`.

## Datasets
### BigCloneBench
BigCloneBench is the dataset for code clone detection task. 
We use the edition from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE). 
You need to download the [datasets](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench/dataset). 
You can download [data.jsonl](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Clone-detection-BigCloneBench/dataset/data.jsonl), [train.txt](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Clone-detection-BigCloneBench/dataset/train.txt), [valid.txt](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Clone-detection-BigCloneBench/dataset/valid.txt) and [test.txt](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Clone-detection-BigCloneBench/dataset/test.txt), then put them into `Clone-detection-BigCloneBench\data\dataset`. 

### CodeSearchNet
CodeSearchNet is the dataset for code summarization task. 
We use the edition from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE). 
The following process is adapted from CodeXGLUE, you can also refer to [this](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text).  

First, you need to download [datasets.zip](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Text/code-to-text/dataset.zip), then put it into `Code-summarization-CodeSearchNet\data`.  
Then
```
unzip dataset.zip
cd dataset
```  

Second, you can download [ruby.zip](https://huggingface.co/datasets/code-search-net/code_search_net/blob/main/data/ruby.zip), [javascript.zip](https://huggingface.co/datasets/code-search-net/code_search_net/blob/main/data/javascript.zip), [java.zip](https://huggingface.co/datasets/code-search-net/code_search_net/blob/main/data/java.zip), [python.zip](https://huggingface.co/datasets/code-search-net/code_search_net/blob/main/data/python.zip), [php.zip](https://huggingface.co/datasets/code-search-net/code_search_net/blob/main/data/php.zip) and [go.zip](https://huggingface.co/datasets/code-search-net/code_search_net/blob/main/data/go.zip) from [huggingface](https://huggingface.co/datasets/code-search-net/code_search_net/tree/main/data).  
Then, put them into `Code-summarization-CodeSearchNet\data\dataset` and unzip them. 
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

## Usages
For code clone detection, you can directly run `trainHyper.py`, it will train, valid and test the pre-trained models with HGAdapter inserted. 
You can modify `pretrain_model_name_or_path` in `trainHyper.py` to choose the pre-trained model, such as 
```
pretrain_model_name_or_path = "codebert-base"
```
The `"codebert-base"` is CodeBERT, you can also choose `"graphcodebert-base"`, i.e., GraphCodeBERT.  
You can also download pre-trained models into `pre_train_models\`, modify the `pretrain_model_name_or_path` to the corresponding path.  

For code summarization task, you can directly run `trainCodellama.py`, it will train, valid and test the Codellama with HGAdapter inserted.  
You can directly run `trainHyper.py`, it will train, valid and test the CodeBERT with HGAdapter inserted.  
You can also download `"CodeLlama-7b"` and `"codebert-base"` into `pre_train_models\`, modify the `pretrain_model_name_or_path` to the corresponding path.  

After the run is complete, the model will be saved to the `work_dir`, and the results will be outputted and saved as a CSV in `work_dir`.  
The implementation of the HGAdapter is located in the `models` folder.  
You can manually modify hyper-parameters such as `batch_size`, `num_epoch` in trainXXX.py.  
Due to time constraints, we have not prepared a user-friendly parameter command for running the code. We will optimize the code to make it easier for users to run directly in future.  
