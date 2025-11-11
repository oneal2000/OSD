## 仓库使用指导

**此github仓库仅用于ANN课程项目开发及提交**
**11.10发现我的requirements好像存在版本冲突，所以直接使用/data-share/miniconda3/envs/re_z环境先进行实验，最后所有实验做完了我们再解决环境版本的问题**
**先下载代码，然后我向你们的文件夹下面复制要用的数据就行了**

### dpr及open-domain qa数据集下载
参考PRAG仓库说明：  
https://github.com/oneal2000/PRAG   

### kilt数据集下载及处理
KILT官方github仓库  
https://github.com/facebookresearch/KILT   
通过官方github仓库下载并处理kilt数据集，可以得到一个kilt.jsonl文件  
然后使用pre.py对其进行处理得到一个可以用来retrieve的简化的jsonl文件kilt_pre.jsonl文件

### elasticsearch来建立索引
参考PRAG仓库说明  
使用的elasticsearch版本是8.15.0  
处理dpr数据集时，使用prep_elastic_dpr.py建立索引  
```
python prep_elastic.py --data_path data/dpr/psgs_w100.tsv --index_name wiki
```
处理kilt数据集时，使用prep_elastic_kilt.py建立索引  
```
python prep_elastic.py --data_path kilt_pre.jsonl --index_name kilt
```

### 检索代码运行
处理dpr数据集时，使用src/retrieve_dpr.py进行检索  
```
python src/retrieve_dpr.py --dataset <dataset_name> --data_path data_dpr/<dataset_name>/ --topk 3 --sample 300
```
最后会在data_ret_dpr/<dataset_name>/目录下生成检索结果文件，同时会在all_docs_dpr.json文件中保存dpr中所有被检索到的文档。    
处理kilt数据集时，使用src/retrieve_kilt.py进行检索  
```
python src/retrieve_kilt.py --dataset <dataset_name> --data_path data_kilt/<dataset_name>/ --topk 3 --sample 300
```
最后会在data_ret_kilt/<dataset_name>/目录下生成检索结果文件，同时会在all_docs_kilt.json文件中保存kilt中所有被检索到的文档。    

### 数据增强
这个数据增强的代码写的不是很好，默认是在dpr和qa上进行增强，如果想要在kilt的另外两个数据集上增强，就要首先要把代码里面的
```
INPUT_FILE = os.path.join(ROOT_DIR, "all_docs_dpr.json")
OUTPUT_FILE = os.path.join(ROOT_DIR, "doc_aug", "dpr_3.json")
```
切换成
```
INPUT_FILE = os.path.join(ROOT_DIR, "all_docs_kilt.json")
OUTPUT_FILE = os.path.join(ROOT_DIR, "doc_aug", "kilt_3.json")
```
**注意，如果扩展了新的数据集，比如strategyqa，可以修改上述文件路径单独为strategy进行检索和增强，比如创建一个data_starategy文件夹和一个all_docs_strategyqa就行，不需要把之前所有的数据再检索或者增强一遍，然后之后的所有步骤对这个数据集进行路径的特判即可，我们先跑出来结果再把代码改的可读和规范**   

**注意，扩展了新的数据集之后这里面读数据集的函数可能要根据数据集的名称和特点重写**  


### task_LoRA生成

使用src/encode_task.py进行task_LoRA的生成

**注意，这里有可能跑出来效果不好，但是很有可能是因为LoRA初始化参数碰巧不是很好，我这边有1b模型初始化效果很好的baseweight，但是3b模型依旧可能遇到这个问题，所以跑完之后先在inference中跑一下FT_LLM_weak方法，如果效果比LLM_direct差就重开，这个跑一次是比较快的，如果5次还烂那可能有点问题，可以在群里讨论**

```
python src/encode_task.py --model_name llama3.2-1b-instruct --task_type <task_type>
```
注意，我建议里面其他的参数全部按照我的默认设置，如有需要之后再改


### doc_LoRA生成

使用src/encode_doc.py进行doc_LoRA的生成

**注意，我们这个阶段所有task_LoRA_type都选择weak！！！**

这里的方法是先把task_LoRA加载到base model上生成一个新的base model，然后再用这个新的base model去生成doc_LoRA

```
python src/encode_doc.py --model_name llama3.2-1b-instruct --dataset <dataset_name> --task_type <task_type>
```

可以在这里修改epoch和learning rate等参数，别的参数先不改了


### baseline怎么跑

#### LLM_direct

直接在src/inference.py中，选择inference_method="LLM_direct"即可，其他参数根据需要修改。

#### RAG

直接在src/inference.py中，选择inference_method="RAG"即可，其他参数根据需要修改。   

#### PRAG

- 首先，使用src/encode.py对增强后的文档进行编码，生成LoRA，具体命令可参考PRAG仓库说明。  
- 然后，使用src/inference.py中，选择inference_method="PRAG"即可，其他参数根据需要修改。  

#### LLM_FT和RAG_FT

- 首先，使用src/task_FT.py生成LoRA
- 然后，使用src/inference.py中，选择inference_method="FT_LLM"或者"inference_method="FT_RAG"即可，其他参数根据需要修改。

