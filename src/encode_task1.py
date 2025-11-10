import os
import gc
import time
import json
import argparse
import torch
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig, PeftModel
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from typing import Dict, List
from prompt_template import get_prompt, get_prompt_llm, get_prompt_fc, get_prompt_fc_llm, get_prompt_sf, get_prompt_sf_llm
from root_dir_path import ROOT_DIR
from utils import get_model
import numpy as np
import random

seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class TrainingData(Dataset):
    ignored_id = -100

    def __init__(self, origin_dataset, tokenizer, args):
        def _get_question(self, data):
            if "question" in data:
                return data["question"]
            elif "input" in data:
                return data["input"]
            else:
                raise ValueError("Neither 'question' nor 'input' found in data!")
        max_length = args.block_size
        self.dataset = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.max_raw_len = 0
        for data in origin_dataset:

            if args.task_type == "open_domain_qa":
                if args.LoRA_type == "RAG":
                    prompt_ids = prompt_template.get_prompt(
                        tokenizer=tokenizer, 
                        question=_get_question(self, data),
                        passages=data["passages"], 
                        answer=None,
                        with_cot=args.with_cot
                    )
                    # print(tokenizer.decode(prompt_ids))
                else:
                    prompt_ids = prompt_template.get_prompt_llm(
                        tokenizer=tokenizer, 
                        question=_get_question(self, data),
                        answer=None,
                        with_cot=args.with_cot
                    )
                    # print(tokenizer.decode(prompt_ids))
            elif args.task_type == "fact_checking":
                if args.LoRA_type == "RAG":
                    prompt_ids = prompt_template.get_prompt_fc(
                        tokenizer=tokenizer, 
                        input=data["input"], 
                        passages=data["passages"],
                        output=None
                    )
                    # print(tokenizer.decode(prompt_ids))
                else:
                    prompt_ids = prompt_template.get_prompt_fc_llm(
                        tokenizer=tokenizer, 
                        input=data["input"],
                        output=None
                    )
            elif args.task_type == "slot_filling":
                if args.LoRA_type == "RAG":
                    prompt_ids = prompt_template.get_prompt_sf(
                        tokenizer=tokenizer, 
                        input=data["input"], 
                        template_question=data["template_question"],
                        passages=data["passages"],
                        output=None
                    )
                    # print(tokenizer.decode(prompt_ids))
                else:
                    prompt_ids = prompt_template.get_prompt_sf_llm(
                        tokenizer=tokenizer, 
                        input=data["input"],
                        template_question=data["template_question"],
                        output=None
                    )

            if args.dataset== "hotpotqa" or args.dataset == "fever":
                answer = data["answer"]
                # print(f"Answer: {answer}")
                answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                answer_ids.append(tokenizer.eos_token_id)
            else:
                answer = data["answer"][0]
                if not answer.endswith("."):
                    answer += "."
                answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                answer_ids.append(tokenizer.eos_token_id)

            # input
            input_ids = prompt_ids + answer_ids
            raw_len = len(input_ids)
            if raw_len > self.max_raw_len:
                self.max_raw_len = raw_len
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
            pad_length = max_length - len(input_ids)

            # attention mask
            attention_mask = [1] * len(input_ids) + [0] * pad_length

            # label
            labels = [self.ignored_id] * len(prompt_ids) + answer_ids # -100 for prompt in label
            if len(labels) > max_length:
                labels = labels[:max_length]
            labels += [self.ignored_id] * (max_length - len(labels))
            input_ids += [pad_token_id] * pad_length

            self.dataset.append({
                "input_ids": input_ids, 
                "labels": labels, 
                "attention_mask": attention_mask,
            })
        self.total_len = len(self.dataset)

        print(f"max_raw_len: {self.max_raw_len}")
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx) -> Dict[str, list]:
        return self.dataset[idx]


class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, examples: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple(
            map(lambda x: [example[x] for example in examples], ["input_ids", "labels", "attention_mask"])
        )
        return {
            "input_ids": torch.tensor(input_ids).to(self.device),
            "labels": torch.tensor(labels).to(self.device),
            "attention_mask": torch.tensor(attention_mask).to(self.device),
        }


def get_train_data(data, tokenizer, task_type, args):
    psg = data["passage"]
    task_list = data.get("task", [])

    prompt_ids = []

    for t in task_list:
        if t["type"] != task_type:
            continue
        for item in t["data"]:
            input = item["input"]
            output = item["output"]
            if task_type == "slot_filling":
                template_question = item["template_question"]
                ids_sf_psg = get_prompt_sf(tokenizer, input, template_question, [psg], output)
                ids_sf_no_psg = get_prompt_sf_llm(tokenizer, input, template_question, output)
                # print(tokenizer.decode(ids_sf_psg, skip_special_tokens=True))
                # print("-----")
                # print(tokenizer.decode(ids_sf_no_psg, skip_special_tokens=True))
                # print("=========")
                prompt_ids.append(ids_sf_psg)
                prompt_ids.append(ids_sf_no_psg)
            elif task_type == "open_domain_qa":
                question = item["input"]
                answer = item["output"]
                full_answer = item["full_answer"]
                ids_qa_psg = get_prompt(tokenizer, question, [psg], answer if not args.with_cot else full_answer, args.with_cot)
                ids_qa_no_psg = get_prompt_llm(tokenizer, question, None, answer if not args.with_cot else full_answer, args.with_cot)
                prompt_ids.append(ids_qa_psg)
                prompt_ids.append(ids_qa_no_psg)
                print(tokenizer.decode(ids_qa_psg, skip_special_tokens=True))
                print("-----")
                print(tokenizer.decode(ids_qa_no_psg, skip_special_tokens=True))
                print("=========")
            elif task_type == "fact_checking":
                claim = item["input"]
                label = item["output"]
                ids_cf_psg = get_prompt_fc(tokenizer, claim, [psg], label)
                ids_cf_no_psg = get_prompt_fc_llm(tokenizer, claim, label)
                # print(tokenizer.decode(ids_cf_psg, skip_special_tokens=True))
                # print("-----")
                # print(tokenizer.decode(ids_cf_no_psg, skip_special_tokens=True))
                # print("=========")
                prompt_ids.append(ids_cf_psg)
                prompt_ids.append(ids_cf_no_psg)

    return prompt_ids


def train(model, prompt_ids, tokenizer, args, init_path, save_path):
    train_data = TrainingData(prompt_ids, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=True,
    )
    model = PeftModel.from_pretrained(model, init_path, is_trainable=True)
    model.is_parallelizable = True
    model.model_parallel = True
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)

    for epoch in range(args.num_train_epochs):
        epoch_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        for step, batch in enumerate(epoch_bar):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    model = model.unload()
    torch.cuda.empty_cache()
    gc.collect()
    return model

def main(args):
    input_file = os.path.join(ROOT_DIR, "doc_aug", "sampled_1500.json")
    with open(input_file, "r") as f:
        input_data = json.load(f)
    
    print(f"Processing {len(input_data)} data for training")

    model, tokenizer, _generation_config = get_model(args.model_name)

    init_path = os.path.join(
        ROOT_DIR,
        "offline_task",
        args.model_name,
        "base_weight"
    )
    if not os.path.exists(os.path.join(init_path, "adapter_model.safetensors")):
        print("No LoRA base weight, creating...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['down_proj', 'gate_proj', 'up_proj'],
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.dropout_rate,
        )
        model = get_peft_model(model, peft_config)
        model.is_parallelizable = True
        model.model_parallel = True
        print(f'Save LoRA base weight to {init_path}')
        os.makedirs(init_path, exist_ok=True)
        model.save_pretrained(init_path)
        time.sleep(2)
        assert os.path.exists(os.path.join(init_path, "adapter_model.safetensors")) 

    save_path = os.path.join(
        ROOT_DIR,
        "offline_task",
        args.model_name,
        args.task_type,
        "LoRA_module"
    )

    os.makedirs(save_path, exist_ok=True)

    model = train(model, prompt_ids, tokenizer, args, init_path, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_type", type = str, choices = ["open_domain_qa", "fact_checking", "slot_filling"], required=True)
    parser.add_argument("--LoRA_type", type = str, default = "LLM", choices = ["RAG", "LLM"])
    parser.add_argument("--with_cot", action="store_true")
    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--sample", type=int, default=-1, help="Number of samples to use, -1 for all samples")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--block_size", type=int, default=1800)
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()
    print(args)
    main(args)