# 2WQA
python src/encode_hard.py \
    --model_name llama3.2-3b-instruct \
    --dataset 2wikimultihopqa \
    --task_type opne_domain_qa \
    --with_cot

# HQA
python src/encode_hard.py \
    --model_name llama3.2-3b-instruct \
    --dataset hotpotqa \
    --task_type opne_domain_qa \
    --with_cot

# CWQ
python src/encode_hard.py \
    --model_name llama3.2-3b-instruct \
    --dataset complexwebquestions \
    --task_type opne_domain_qa 

# PopQA
python src/encode_hard.py \
    --model_name llama3.2-3b-instruct \
    --dataset popqa \
    --task_type opne_domain_qa \
    --epoch 2

# FEVER
python src/encode_hard.py \
    --model_name llama3.2-3b-instruct \
    --dataset fever \
    --task_type fact_checking \
    --learning_rate 1e-4

# Zero Shot RE
python src/encode_hard.py \
    --model_name llama3.2-3b-instruct \
    --dataset zero_shot_re \
    --task_type slot_filling \
    --epoch 2 \
    --learning_rate 5e-4

# WoW
python src/encode_hard.py \
    --model_name llama3.2-3b-instruct \
    --dataset wow \
    --task_type dialogue \
    --epoch 2

# PubMedQA
python src/encode_hard.py \
    --model_name llama3.2-3b-instruct \
    --dataset pubmedqa \
    --task_type med_verify