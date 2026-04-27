# this script is for training task LoRA with 1500 samples from datasets of each task
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
from utils import get_model

seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class TrainingData(Dataset):
    ignored_id = -100

    def __init__(self, origin_dataset, tokenizer, args):
        def _get_question(data):
            if "question" in data:
                return data["question"]
            elif "input" in data:
                return data["input"]

        max_length = args.block_size
        self.dataset = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.max_raw_len = 0

        for data in origin_dataset:
            if args.task_type == "open_domain_qa":
                question = _get_question(data)
                if args.LoRA_type == "RAG":
                    prompt_ids = prompt_template.get_prompt(
                        tokenizer=tokenizer, 
                        question=question,
                        passages=data["passages"], 
                        answer=None,
                        with_cot=args.with_cot
                    )
                else:
                    prompt_ids = prompt_template.get_prompt(
                        tokenizer=tokenizer, 
                        question=question,
                        passages=None, 
                        answer=None,
                        with_cot=args.with_cot
                    )
            elif args.task_type == "fact_checking":
                if args.LoRA_type == "RAG":
                    prompt_ids = prompt_template.get_prompt_fc(
                        tokenizer=tokenizer, 
                        input=data["input"], 
                        passages=data["passages"],
                        output=None
                    )
                else: # LLM
                    prompt_ids = prompt_template.get_prompt_fc(
                        tokenizer=tokenizer, 
                        input=data["input"], 
                        passages=None,
                        output=None
                    )
            elif args.task_type == "dialogue":
                if args.LoRA_type == "RAG":
                    prompt_ids = prompt_template.get_prompt_dialogue(
                        tokenizer=tokenizer, 
                        input=data["input"], 
                        passages=data["passages"],
                        output=None
                    )
                else: # LLM
                    prompt_ids = prompt_template.get_prompt_dialogue(
                        tokenizer=tokenizer, 
                        input=data["input"], 
                        passages=None,
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
                else: # LLM
                    prompt_ids = prompt_template.get_prompt_sf(
                        tokenizer=tokenizer, 
                        input=data["input"], 
                        template_question=data["template_question"],
                        passages=None,
                        output=None
                    )
            elif args.task_type == "med_verify":
                question = data["question"]
                if args.LoRA_type == "RAG":
                    prompt_ids = prompt_template.get_prompt_pubmedqa(
                        tokenizer=tokenizer,
                        question=question,
                        passages=data["passages"],
                        answer=None
                    )
                else:  # LLM
                    prompt_ids = prompt_template.get_prompt_pubmedqa_llm(
                        tokenizer=tokenizer,
                        question=question,
                        answer=None
                    )

            answer = data["answer"]
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            answer_ids.append(tokenizer.eos_token_id)

            input_ids = prompt_ids + answer_ids
            # print(tokenizer.decode(input_ids,skip_special_tokens=True))
            raw_len = len(input_ids)
            if raw_len > self.max_raw_len:
                self.max_raw_len = raw_len
            
            # Truncate if longer than max_length
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]

            labels = [self.ignored_id] * len(prompt_ids) + answer_ids # ignore prompt part in loss
            if len(labels) > max_length:
                labels = labels[:max_length]
            
            pad_length = max_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_length
            input_ids += [pad_token_id] * pad_length
            labels += [self.ignored_id] * (max_length - len(labels))

            self.dataset.append({
                "input_ids": input_ids, 
                "labels": labels, 
                "attention_mask": attention_mask,
            })
            
        self.total_len = len(self.dataset)
        print(f"Processed {self.total_len} samples. Max sequence length: {self.max_raw_len}")
    
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
            "input_ids": torch.tensor(input_ids, dtype=torch.long).to(self.device),
            "labels": torch.tensor(labels, dtype=torch.long).to(self.device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long).to(self.device),
        }

def main(args):
    input_file = os.path.join(ROOT_DIR, "doc_aug", "pub_3.json")
    with open(input_file, "r") as f:
        input_data = json.load(f)

    training_samples = []
    for entry in input_data:
        passage = entry["passage"]
        for task in entry["task"]:
            if task["type"] == args.task_type:
                # The 'data' field can be a dict or a list of dicts
                task_data_list = task["data"] if isinstance(task["data"], list) else [task["data"]]
                
                for item in task_data_list:
                    if args.task_type == "med_verify":
                        sample = {
                            "passages": [passage],
                            "question": item["question"],
                            "answer": item["answer"],  
                        }
                    else:
                        sample = {
                            "passages": [passage],
                            "input": item["input"],
                        }
                        if args.task_type == "open_domain_qa":
                            sample["answer"] = item["full_answer"] if args.with_cot else item["output"]
                        elif args.task_type == "slot_filling":
                            sample["answer"] = item["output"]
                            sample["template_question"] = item["template_question"]
                        elif args.task_type == "dialogue":
                            sample["answer"] = item["output"]
                        else: # fact_checking
                            sample["answer"] = item["output"]
                    training_samples.append(sample)

    print(f"Extracted {len(training_samples)} samples for task type: '{args.task_type}'")

    if args.task_type == "fact_checking":
        supports = [s for s in training_samples if s["answer"].strip().upper() == "SUPPORTS"]
        refutes = [s for s in training_samples if s["answer"].strip().upper() == "REFUTES"]
        min_count = min(len(supports), len(refutes))
        random.shuffle(supports)
        random.shuffle(refutes)
        training_samples = supports[:min_count] + refutes[:min_count]
        random.shuffle(training_samples)
        print(f"Balanced fact_checking samples: SUPPORTS={min_count}, REFUTES={min_count}")

    if args.task_type == "med_verify":
        yess = [s for s in training_samples if s["answer"].strip()== "yes"]
        nos = [s for s in training_samples if s["answer"].strip() == "no"]
        min_count = min(len(yess), len(nos))
        training_samples = yess[:min_count] + nos[:min_count]
        print(f"Balanced med_verify samples: YES={min_count}, NO={min_count}")
    
    if args.sample > 0:
        random.shuffle(training_samples)
        training_samples = training_samples[:args.sample]
        print(f"Using a subset of {len(training_samples)} samples.")
    
    model, tokenizer, _ = get_model(args.model_name)
    
    init_path = os.path.join(ROOT_DIR, "offline_task", args.model_name, "base_weight")
    if not os.path.exists(os.path.join(init_path, "adapter_model.safetensors")):
        print("No LoRA base weight found, creating one...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['down_proj', 'gate_proj', 'up_proj'],
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.dropout_rate,
        )
        model = get_peft_model(model, peft_config)
        print(f'Saving LoRA base weight to {init_path}')
        os.makedirs(init_path, exist_ok=True)
        model.save_pretrained(init_path)
        time.sleep(2)

        del model
        torch.cuda.empty_cache()
        gc.collect()
        model, tokenizer, _ = get_model(args.model_name)

    random.shuffle(training_samples)
    split_idx = int(len(training_samples) * 0.9)
    train_split, val_split = training_samples[:split_idx], training_samples[split_idx:]
    
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

    # --- 5. Training Setup ---
    save_path = os.path.join(
        ROOT_DIR, "offline_task", args.model_name, args.task_type, args.LoRA_type
        # f"epoch={args.num_train_epochs}_lr={args.learning_rate}_dropout={args.dropout_rate}"
    )
    os.makedirs(save_path, exist_ok=True)

    model = PeftModel.from_pretrained(model, init_path, is_trainable=True)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    logging_step = 10
    train_losses, val_losses, val_steps = [], [], []

    print(f"Starting training for {args.num_train_epochs} epochs. Saving model to {save_path}")
    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        for step, batch in enumerate(epoch_bar):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            epoch_bar.set_postfix(loss=loss.item())

            # Validation step
            if (step + 1) % logging_step == 0:
                model.eval()
                val_batch_losses = []
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_outputs = model(**val_batch)
                        val_batch_losses.append(val_outputs.loss.item())
                avg_val_loss = np.mean(val_batch_losses)
                val_losses.append(avg_val_loss)
                val_steps.append(len(train_losses) - 1)
                print(f"\nEpoch {epoch+1}, Step {step+1}: Train Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}")
                model.train()

    model.save_pretrained(save_path)

    # Plot and save the loss curve
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_steps, val_losses, label="Validation Loss", marker='o')
    plt.title("Training & Validation Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    
    model = model.unload()
    torch.cuda.empty_cache()
    gc.collect()
    print("Training finished and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_type", type=str, choices=["open_domain_qa", "fact_checking", "slot_filling", "dialogue", "med_verify"], required=True)
    parser.add_argument("--LoRA_type", type=str, default="LLM", choices=["RAG", "LLM"])
    parser.add_argument("--with_cot", action="store_true")
    
    # Training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--block_size", type=int, default=300)
    parser.add_argument("--sample", type=int, default=-1)
    
    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    args = parser.parse_args()
    print("Training with the following arguments:")
    print(json.dumps(vars(args), indent=4))
    
    main(args)