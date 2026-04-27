# 2WQA
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset 2wikimultihopqa \
    --task_type opne_domain_qa \
    --with_cot

# HQA
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset hotpotqa \
    --task_type opne_domain_qa \
    --with_cot

# CWQ
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset complexwebquestions \
    --task_type opne_domain_qa 

# PopQA
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset popqa \
    --task_type opne_domain_qa \
    --epoch 2

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
    --epoch 2

# WoW
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset wow \
    --task_type dialogue \
    --epoch 2

# PubMedQA
python src/encode_doc.py \
    --model_name llama3.2-1b-instruct \
    --dataset pubmedqa \
    --task_type med_verify