# 2WQA
python src/inference.py \
    --model_name llama3.1-8b-instruct \
    --dataset 2wikimultihopqa \
    --task_type opne_domain_qa \
    --lambda_orth 0.2 \
    --with_cot \
    --inference_method D-PRAG \
    --max_new_tokens 128

# HQA
python src/inference.py \
    --model_name llama3.1-8b-instruct \
    --dataset hotpotqa \
    --task_type opne_domain_qa \
    --lambda_orth 0.2 \
    --with_cot \
    --inference_method D-PRAG \
    --max_new_tokens 128

# CWQ
python src/inference.py \
    --model_name llama3.1-8b-instruct \
    --dataset complexwebquestions \
    --task_type opne_domain_qa \
    --lambda_orth 0.2 \
    --inference_method D-PRAG \
    --max_new_tokens 20

# PopQA
python src/inference.py \
    --model_name llama3.1-8b-instruct \
    --dataset popqa \
    --task_type opne_domain_qa \
    --epoch 2 \
    --lambda_orth 0.2 \
    --inference_method D-PRAG \
    --max_new_tokens 20

# FEVER
python src/inference.py \
    --model_name llama3.1-8b-instruct \
    --dataset fever \
    --task_type fact_checking \
    --learning_rate 5e-5 \
    --lambda_orth 0.2 \
    --inference_method D-PRAG \
    --max_new_tokens 20

# Zero Shot RE
python src/inference.py \
    --model_name llama3.1-8b-instruct \
    --dataset zero_shot_re \
    --task_type slot_filling \
    --epoch 2 \
    --learning_rate 5e-4 \
    --lambda_orth 0.2 \
    --inference_method D-PRAG \
    --max_new_tokens 20

# WoW
python src/inference.py \
    --model_name llama3.1-8b-instruct \
    --dataset wow \
    --task_type dialogue \
    --epoch 2 \
    --lambda_orth 0.2 \
    --inference_method D-PRAG \
    --max_new_tokens 32

# PubMedQA
python src/inference.py \
    --model_name llama3.1-8b-instruct \
    --dataset pubmedqa \
    --task_type med_verify \
    --lambda_orth 0.2 \
    --inference_method D-PRAG \
    --max_new_tokens 20