### 环境创建

```bash
conda create -n dprag python=3.10
conda activate dprag
pip install -r requirements.txt
```

### 数据集下载

有关 KILT 相关数据集获取参考 [KILT](https://github.com/facebookresearch/KILT)，按照官方仓库的分段文件处理为`kilt.jsonl`之后使用`src/pre.py`进行处理得到`kilt_pre.jsonl`。  
有关 PubmedQA 数据集获取参考 [PubmedQA](https://github.com/pubmedqa/pubmedqa)，经过数据处理可以得到`pubmed.jsonl`，此数据集可见链接[pubmed.jsonl](https://www.dropbox.com/scl/fi/u0ne41rznvy5b3kchhxx7/pubmed.jsonl?rlkey=fk0bqnclk2eyyhg8oz5arx81d&e=1&st=ub9dp3h9&dl=0)

### ElasticSearch

为DPR准备ElasticSearch索引
```bash
cd data_dpr
wget -O elasticsearch-8.15.0.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-8.15.0.tar.gz
rm elasticsearch-8.15.0.tar.gz 
cd elasticsearch-8.15.0
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ../..
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
```
为KILT准备索引，下载elasticsearch-8.15.0并后台运行的方法同上，不再赘述。
```bash
python prep_elastic_kilt.py --data_path kilt_pre.jsonl --index_name kilt
```
为PubMed准备索引，下载elasticsearch-8.15.0并后台运行的方法同上，不再赘述。
```bash
python prep_elastic_med.py --data_path pubmed.jsonl --index_name med
```

下载数据集：  
可按照报告中给出的数据集链接直接下载。  
将下载好的数据集按照如下方法分类放置：  
- data_dpr:2wikimultihopqa,popqa,hotpotqa,complexwebquestions
- data_kilt:fever,zeroshot_re,wow
- data_med:pubmedqa

### 代码运行

首先修改`src/root_dir_path`为此文件夹的目录路径。

#### 数据检索
在创建好ElasticSearch索引并下载好数据集后，运行以下命令进行数据检索：
```bash
python src/retrieve_dpr.py --dataset <dataset_name> --data_path data_dpr/<dataset_name>/ --topk 3 --sample 300

python src/retrieve_kilt.py --dataset <dataset_name> --data_path data_dpr/<dataset_name>/ --topk 3 --sample 300

python src/retrieve_med.py --dataset <dataset_name> --data_path data_dpr/<dataset_name>/ --topk 3 --sample 300
```

#### 数据增强
```bash
python src/augment.py
```
默认使用`llama3-8b-instruct`进行数据增强，如需更换模型，请在`src/augment.py`中修改`model_name`并在`src/utils.py`中添加对应模型的加载方法。

#### PRAG训练
```bash
python src/encode.py --model_name <model_name> --dataset <dataset_name> --task_type <task_type>
```
默认学习率为1e-3，epoch为1，部分数据集的参数有改动，设置可见说明部分。  

#### task_lora训练
```bash
python src/encode_task.py --model_name <model_name> --task_type <task_type>
```
默认学习率为1e-4，epoch为1，部分任务类型有改动，设置可见说明部分。

#### doc_lora训练
```bash
python src/encode_doc.py --model_name <model_name> --dataset <dataset_name> --task_type <task_type>
```
默认学习率为1e-3，epoch为1，lambda_orth为0.1，部分数据集的参数有改动，设置可见说明部分。

#### 推理
```bash
python src/inference.py --model_name <model_name> --dataset <dataset_name>  --task_type <task_type> --inference_method <method> --max_new_tokens <num>
```
其中method包括`LLM_direct`, `RAG`, `PRAG`, `D-PRAG`, `D-PRAG-combine`。`max_new_tokens`默认使用20，epoch和学习率与PRAG和D-PRAG encode阶段选择的参数保持一致，部分数据集的参数改动在说明部分。

### 说明

- 在训练LoRA时为排除参数影响，统一数据集的PRAG和D-PRAG的学习率等参数均保持一致，后面除lambda_orth以及inference和task_lora训练阶段的参数其余参数均在PRAG和D-PRAG中保持一致。
- 2wikimultihopqa和hotpotqa在LLM_direct，RAG，PRAG和PRAG-combine推理时采取PRAG论文中with_cot的方法，D-PRAG及D-PRAG-combine在1b和3b模型上不采用with_cot方法，在8b模型上采用with_cot方法。
- 训练过程参数有改动的数据集：
    - hotpotqa在8b模型上训练doc_lora时选择lambda_orth为0.05
    - popqa在3种规模的模型上训练doc_lora的epoch均为2
    - fact_checking的task_lora在1b模型上选择epoch为3，在3b模型上选择学习率为8e-5
    - fever数据集在3b模型上训练doc_lora时选择lambda_orth为0.05，学习率选择1e-4，在8b模型上选择学习率为5e-5
    - zeroshot_re的训练epoch选择2，在3b和8b模型上学习率选择5e-4
    - dialogue-generation的task_lora在1b和3b模型上选择epoch=2
    - wow在训练doc_lora时三种模型上的epoch均为2
    - pubmedqa在3b模型上训练doc_lora时选择epoch为2
- 推理过程参数有改动的数据集：
    - 2wikimultihopqa和hotpotqa的max_new_tokens选择128
    - wow的max_new_tokens选择32
- 如您在复现实验结果时遇到任何问题，请随时与我们组的任何一位同学联系，谢谢！
