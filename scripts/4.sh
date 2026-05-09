CUDA_VISIBLE_DEVICES=1 python src/encode_doc.py --model_name llama3.2-3b-instruct --dataset 2wikimultihopqa --task_type open_domain_qa --with_cot
CUDA_VISIBLE_DEVICES=1 python src/encode_doc.py --model_name llama3.2-3b-instruct --dataset hotpotqa --task_type open_domain_qa --with_cot
CUDA_VISIBLE_DEVICES=1 python src/encode_doc.py --model_name llama3.2-3b-instruct --dataset wow --task_type dialogue --num_train_epochs 2
CUDA_VISIBLE_DEVICES=1 python src/encode_doc.py --model_name llama3.2-3b-instruct --dataset pubmedqa --task_type med_verify