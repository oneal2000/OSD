#!/bin/bash

# 定义数据集列表和任务类型
declare -A datasets
datasets=(
    # ["fever"]="fact_checking --max_new_tokens 20 --learning_rate 5e-5"
    # ["zeroshot_re"]="slot_filling --num_train_epochs 2 --max_new_tokens 20 --learning_rate 5e-4"
    # ["wow"]="dialogue --num_train_epochs 2 --max_new_tokens 32"
    ["pubmedqa"]="med_verify --max_new_tokens 20"
)

# 定义要遍历的 inference_method
methods=("D-PRAG" "D-PRAG-combine")

# 定义 doc_num 遍历值
doc_nums=(1 3 5 7 10)

# 遍历每个数据集
for dataset in "${!datasets[@]}"; do
    task_params=${datasets[$dataset]}

    for method in "${methods[@]}"; do
        for doc_num in "${doc_nums[@]}"; do
            echo "Running dataset: $dataset, method: $method, doc_num: $doc_num"
            CUDA_VISIBLE_DEVICES=3 python src/inference.py \
                --model_name llama3.1-8b-instruct\
                --dataset $dataset \
                --task_type $(echo $task_params | awk '{print $1}') \
                $(echo $task_params | cut -d' ' -f2-) \
                --inference_method $method \
                --doc_num $doc_num \
                --lambda_orth 10
        done
    done
done