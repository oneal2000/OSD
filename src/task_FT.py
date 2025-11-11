# this file is to train baseline FT_LLM and FT_RAG LoRA
# the training process of FT_LLM uses only input-output pairs of specific dataset
# the training process of FT_RAG uses input-output pairs and retrieved passages of specific dataset
import os
import gc
import json
import time
import numpy as np
import random
import argparse
import torch
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig, PeftModel
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from transformers import DefaultDataCollator
from typing import Dict, List

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, load_data

seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class TrainingData(Dataset):
    ignored_id = -100

    def __init__(self, origin_dataset, tokenizer, args):
        if args.task_type == "fact_checking":
            special_tokens = {"additional_special_tokens": ["<SUPPORTS>", "<REFUTES>"]}
            tokenizer.add_special_tokens(special_tokens)
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
    

def main(args):
    model, tokenizer, _generation_config = get_model(args.model_name)
    data_dir = os.path.join(ROOT_DIR, "FT_data", args.dataset)
    init_adapter_path = os.path.join(ROOT_DIR, "offline_FT", args.model_name, "base_weight")

    data_list = load_data(None, None, None, data_dir=data_dir)
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    if not os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")):
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
        os.makedirs(init_adapter_path, exist_ok=True)
        model.save_pretrained(init_adapter_path)
        time.sleep(2)

    for filename, fulldata in data_list:
        filename = filename.split('.')[0] 
        print(f"### Solving {filename} ###")

        fulldata = fulldata if args.sample == -1 else fulldata[:args.sample]

        train_split, val_split = fulldata[:900], fulldata[-100:] # 9:1 split
        train_data = TrainingData(train_split, tokenizer, args)
        val_data = TrainingData(val_split, tokenizer, args)

        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.per_device_train_batch_size,
            collate_fn=TrainingDataCollator(tokenizer, model.device),
            shuffle=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.per_device_train_batch_size,
            collate_fn=TrainingDataCollator(tokenizer, model.device),
            shuffle=False,
        )

        save_path = os.path.join(
            ROOT_DIR, 
            "offline_FT", 
            args.model_name,
            args.dataset,
            args.LoRA_type,
            f"batch={args.per_device_train_batch_size}_epoch={args.num_train_epochs}_lr={args.learning_rate}_dropout={args.dropout_rate}",
            filename
        )
        os.makedirs(save_path, exist_ok=True)

        model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
        model.is_parallelizable = True
        model.model_parallel = True

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

        logging_step = 10
        train_losses, val_losses, val_steps = [], [], []

        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")):
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                if step % logging_step == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Train Loss: {loss.item():.4f}")

                    model.eval()
                    val_batch_losses = []
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            val_outputs = model(**val_batch)
                            val_batch_losses.append(val_outputs.loss.item())
                    avg_val_loss = np.mean(val_batch_losses)
                    val_losses.append(avg_val_loss)
                    val_steps.append(len(train_losses)-1)
                    print(f"Epoch {epoch+1}, Step {step}, Validation Loss: {avg_val_loss:.4f}")
                    model.train()

        model.save_pretrained(save_path)
        with open(os.path.join(save_path, "training_config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)

        plt.figure(dpi=300)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_steps, val_losses, label="Validation Loss")
        plt.title("Training & Validation Loss Curve")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(save_path, "loss.png"))

        model = model.unload()
        torch.cuda.empty_cache()
        gc.collect()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task_type", type = str, choices = ["open_domain_qa", "fact_checking", "slot_filling"], required=True)
    parser.add_argument("--LoRA_type", type = str, default = "RAG", choices = ["RAG", "LLM"])
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
    main(args)