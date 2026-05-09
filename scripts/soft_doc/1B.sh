# 2WQA
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset 2wikimultihopqa \
    --task_type open_domain_qa \
    --with_cot \
    --block_size 1500

# HQA
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset hotpotqa \
    --task_type open_domain_qa \
    --with_cot \
    --block_size 1500

# CWQ
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset complexwebquestions \
    --task_type open_domain_qa 

# PopQA
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset popqa \
    --task_type open_domain_qa \
    --num_train_epochs 2

# FEVER
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset fever \
    --task_type fact_checking

# Zero Shot RE
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset zero_shot_re \
    --task_type slot_filling \
    --num_train_epochs 2

# WoW
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset wow \
    --task_type dialogue \
    --num_train_epochs 2

# PubMedQA
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset pubmedqa \
    --task_type med_verify