#!/bin/bash

# 定义数据集列表和任务类型
declare -A datasets
datasets=(
    # ["2wikimultihopqa"]="open_domain_qa --with_cot --max_new_tokens 128"
    ["hotpotqa"]="open_domain_qa --with_cot --max_new_tokens 128"
    # ["complexwebquestions"]="open_domain_qa --max_new_tokens 20"
    # ["popqa"]="open_domain_qa --num_train_epochs 2 --max_new_tokens 20"
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
            CUDA_VISIBLE_DEVICES=6 python src/inference.py \
                --model_name llama3.2-1b-instruct \
                --dataset $dataset \
                --task_type $(echo $task_params | awk '{print $1}') \
                $(echo $task_params | cut -d' ' -f2-) \
                --inference_method $method \
                --doc_num $doc_num \
                --lambda_orth 10
        done
    done
done