# D-PRAG: Decoupling Knowledge and Task Subspaces for Composable Parametric Retrieval Augmented Generation

**Welcome to the Official Repository of Decoupling Knowledge and Task Subspaces for Composable Parametric
Retrieval Augmented Generation**   


This repository contains the code, datasets models used in our paper.  
If you find our project helpful, we would sincerely appreciate it if you could give us a star!   

Orthorgonal Subspace Decomposition for Parametric Retrieval Augmented Generation (D-PRAG) is a novel framework that decouples the knowledge and task subspaces in PRAG, which explicitly decouple parameterized external memory into two functionally distinct subspaces: a task subspace, captured by a shared Task LoRA that models generic task execution, and a knowledge subspace, captured by document-specific Knowledge LoRAs that encode factual content.   

This repository includes the code for training and evaluating D-PRAG. We also provide the preprocessed datasets used in our experiments.

## Reproduce Paper Results

### Install Environment

```bash
conda create -n dprag python=3.10
conda activate dprag
pip install -r requirements.txt
```
Please change the `ROOT_DIR` variable in `src/root_dir_path.py` to the folder address where you store D-PRAG.


### Data Augmentation and Preprocessing

You can directly use the processed datasets we rovide in `data_ret.tar.gz` and `data_aug.tar.gz`, you can directly extract them to get the datasets for D-PRAG.   
- `data_ret`: it includes the test sets we use for each dataset and the retrieved passages for each test sample. Extract the archive and move all its contents into the D-PRAG home directory.
- `data_aug`: it includes the augmented data for each dataset. 

If you want to process the datasets by yourself, you can follow the instructions below.   

#### Data Preparation

First, you should prepare the corpus and datasets used in our experiments.  

**Corpus:**  

1. Download the Wikipedia dump from the [DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command

```bash
mkdir -p data_dpr
wget -O data_dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data_dpr
gzip -d psgs_w100.tsv.gz
popd
```

2. Follow the instructions in [KILT](https://github.com/facebookresearch/KILT), use the splitting files provided in the official repository to process the data into `kilt.jsonl`, and then use `src/pre.py` to further process it into `kilt_pre.jsonl`.

3. The processed pubmedqa dataset we use is provided in the link [pubmed.jsonl](https://www.dropbox.com/scl/fi/u0ne41rznvy5b3kchhxx7/pubmed.jsonl?rlkey=fk0bqnclk2eyyhg8oz5arx81d&e=1&st=ub9dp3h9&dl=0).   

**Datasets:**   

For 2WQA, HQA, CWQ and PQA, you can follow the instructions in [PRAG](https://github.com/oneal2000/PRAG), and put the downloaded datasets into the `data_dpr` folder.   
For FEVER, Zero Shot RE and Wow, you can follow the instructions in [KILT](https://github.com/facebookresearch/KILT), and put the downloaded `fever-dev-kilt.jsonl`, `structured-zeroshot-dev-kilt.jsonl` and `wow-dev-kilt.jsonl` into the `data_kilt` folder.   
For PubMedQA, you can download the dataset from https://www.dropbox.com/scl/fo/357s89d2vxj9c6t9pljw5/AFdREFA65bJ-zlOj5QGAJlk?rlkey=h2h8qudovwzevllwmw04pzvoz&st=l3p7312b&dl=0 and put `train.jsonl` into the `data_pub` folder.

#### Retrieval

Build the Elasticsearch index for DPR, KILT and PubMed datasets.    
First, you need to download Elasticsearch-8.15.0 and run it in the background. You can use the following commands to do that.    
```bash
wget -O elasticsearch-8.15.0.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-8.15.0.tar.gz
rm elasticsearch-8.15.0.tar.gz 
cd elasticsearch-8.15.0
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ..
```
For DPR, you can use the following command to build the index.    
```bash
python prep_elastic.py --data_path data_dpr/psgs_w100.tsv --index_name wiki
```
For KILT, you can use the following command to build the index.
```bash
python prep_elastic_kilt.py --data_path kilt_pre.jsonl --index_name kilt
```
For PubMedQA, you can use the following command to build the index.
```bash
python prep_elastic_med.py --data_path pubmed.jsonl --index_name med
```

After retrieval, you will get the test sets with its retrieved passages for each dataset in `data_ret_*`. And you will get the whole passage collection in `all_docs_*.json`.    

**Note**: You may encounter some questions when building the Elasticsearch index, you can refer to https://discuss.elastic.co/ for help. It’s very active and provides solutions to most common errors. We hope this resource helps you build the index smoothly and successfully complete your reproduction!

#### Data Augmentation

In Knowledge-Intensive Tasks. For each retrieved passage, we generate a rewrite, and populate the augment field with 12 items (3 QA, 3 fact-checking, 3 slot-filling, and 3 dialogue) and the task field with 4 items (1 QA, 1 fact-checking, 1 slot-filling, and 1 dialogue).   

In this section, you should use the following command to generate the augmented data for each dataset.    
```bash
python augment.py --input_file all_docs_dpr.json --output_file doc_aug/dpr.json

python augment.py --input_file all_docs_kilt.json --output_file doc_aug/kilt.json
```

When you finish the augmentation for DPR and KILT, you can get the augmented data for each passage in `doc_aug/dpr.json` and `doc_aug/kilt.json`.   

**Note**: The preprocessed datasets we provide in `doc_aug.tar.gz` have different structures from the augmented data you get by running `augment.py` on the whole retrieved passages. For convenience, we separate datasets and generate augment and task field specific to each dataset. The preprocessed datasets contain the following:

- DPR: both augment and task fields contain only QA pairs.
- FEVER: only fact-checking pairs are included.
- ZSRE: only slot-filling pairs are included.
- WoW: only dialogue pairs are included.


In Vertical Domains(Medical Verification). For each retrieved passage, we generate a rewrite, and populate the augment field with 3 items (3 Medical Verification) and the task field with 1 item (1 Medical Verification).   
You can use the following command to generate the augmented data for PubMedQA.    
```bash
python augment.py --input_file all_docs_med.json --output_file doc_aug/pub.json
```

### Task LoRA Training
By calling `src/encode_task.py`, you can train the Task LoRA.  The trained Task LoRA will be saved in `output_task`.       
The detailed instructions we use in our experiment for training Task LoRA can be found in [scripts](./scripts/task).   
The training data for Task LoRA is sampled from the augmented data generated in the previous step.   
We sample 1500 instances for knowledge-intensive tasks and 900 instances for vertical domain tasks.   
The data we use in our experiment is provided in `data_aug.tar.gz`, you can directly extract it to get the training data for Task LoRA.   


### Doc LoRA Training

#### Soft Orthogonality Constraint
By calling `src/encode_doc.py`, you can train the Doc LoRA with soft orthogonality constraint. The trained Doc LoRA will be saved in `offline_doc/…/lambda=x`.   
The detailed instructions we use in our experiment for training Doc LoRA with soft orthogonality constraint can be found in [scripts](./scripts/soft_doc).  

#### Hard Orthogonality Constraint
By calling `src/encode_hard.py`, you can train the Doc LoRA with soft orthogonality constraint. The trained Doc LoRA will be saved in `offline_doc/…/hard_orth`.   
The detailed instructions we use in our experiment for training Doc LoRA with soft orthogonality constraint can be found in [scripts](./scripts/soft_doc).  


### Inference
By calling `src/inference.py`, you can evaluate the performance of D-PRAG, D-PRAG-hard and other baselines. The results will be saved in `output`.   
The detailed instructions we use in our experiment for inference can be found in [scripts](./scripts/infer). You need to change the `doc_num` in the instruction to specify the number of retrieved passages you want to use for inference and `iinference_method` to specify the inference method you want to use.   


If you have any questions about the code or encounter any issues when reproducing our results, please feel free to open an issue in this repository. We will do our best to help you resolve the problems and successfully reproduce our results!