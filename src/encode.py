# This file is to train baseline PRAG LoRA
import os
import gc
import time
import argparse
import torch
import json
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig, PeftModel
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from typing import Dict, List

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, load_data

from prompt_template import get_prompt, get_prompt_llm, get_prompt_fc, get_prompt_sf, get_prompt_fc_llm, get_prompt_sf_llm

import numpy as np
import random

seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class TrainingData(Dataset):
    ignored_id = -100

    def __init__(self, prompt_ids, tokenizer, max_length=3000):
        self.max_length = max_length
        self.dataset = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for input_ids in prompt_ids:
            labels = input_ids.copy()
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
            input_ids += [pad_token_id] * (max_length - len(input_ids))
            labels += [self.ignored_id] * (max_length - len(labels))
            self.dataset.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })
        self.total_len = len(self.dataset)
    
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
    

def get_train_data(augments, tokenizer, args):
    prompt_ids = []
    psg = augments["passage"]
    for aug in augments["augment"]:
        rew = aug["rewrite"]
        qas = aug["qa"]
        fcs = aug["fact_checking"]
        sfs = aug["slot_filling"]
        qpa_cnt = (len(qas) + 1) // 2
        if args.task_type == "open_domain_qa":
            for qid, qa in enumerate(qas):
                if qid < qpa_cnt:
                    for ppp in [psg, rew]:
                        prompt_ids.append(get_prompt(tokenizer, qa["question"], 
                                                        [ppp], 
                                                        qa["answer"] if not args.with_cot else qa["full_answer"], 
                                                        with_cot=args.with_cot))
                else:
                    prompt_ids.append(get_prompt_llm(tokenizer, qa["question"], 
                                                    qa["answer"] if not args.with_cot else qa["full_answer"], 
                                                    with_cot=args.with_cot))
        elif args.task_type == "fact_checking":
            for fid, fc in enumerate(fcs):
                if fid < qpa_cnt:
                    for ppp in [psg, rew]:
                        prompt_ids.append(get_prompt_fc(tokenizer, fc["input"], 
                                                        [ppp], 
                                                        fc["output"]))
                else:
                    prompt_ids.append(get_prompt_fc_llm(tokenizer, fc["input"], 
                                                        fc["output"]))
        elif args.task_type == "slot_filling":
            for sid, sf in enumerate(sfs):
                # print(type(sf["template_question"]), sf["template_question"])
                if sid < qpa_cnt:
                    for ppp in [psg, rew]:
                        # print([ppp])
                        prompt_ids.append(get_prompt_sf(tokenizer, sf["input"], 
                                                        sf["template_question"], [ppp],
                                                        sf["output"]))
                else:
                    prompt_ids.append(get_prompt_sf_llm(tokenizer, sf["input"], sf["template_question"],
                                                        sf["output"]))
    return prompt_ids


def train(question, augments, args, model, tokenizer, 
          init_adapter_path, save_path):
    prompt_ids = get_train_data(augments, tokenizer, args)
    train_data = TrainingData(prompt_ids, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=False,
    )
    model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
    model.is_parallelizable = True
    model.model_parallel = True
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
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
    if args.dataset in ["fever", "zeroshot_re", "triviaqa"]:
        data_dir = os.path.join(ROOT_DIR, "data_ret_kilt", args.dataset)
        aug_file = os.path.join(ROOT_DIR, "doc_aug", "kilt_3.json")
    elif args.dataset == "test":
        data_dir = os.path.join(ROOT_DIR, "data_ret_test", args.dataset)
        aug_file = os.path.join(ROOT_DIR, "doc_aug", "test.json")
    else:
        data_dir = os.path.join(ROOT_DIR, "data_ret_dpr", args.dataset)
        aug_file = os.path.join(ROOT_DIR, "doc_aug", "dpr.json")
    data_list = load_data(None, None, None, data_dir=data_dir)

    with open(aug_file, "r", encoding="utf-8") as f:
        aug_data_list = json.load(f)

    aug_map = {item["global_id"]: item["augment"] for item in aug_data_list}

    model, tokenizer, _generation_config = get_model(args.model_name)
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    init_adapter_path = os.path.join(
        ROOT_DIR, 
        "offline_prag", 
        args.model_name, 
        "base_weight",
    )
    if not os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")):
        print("No LoRA base weight, creating...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['down_proj', 'gate_proj', 'up_proj'],
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0, # !!!
        )
        model = get_peft_model(model, peft_config)
        model.is_parallelizable = True
        model.model_parallel = True
        print(f'Save LoRA base weight to {init_adapter_path}')
        os.makedirs(init_adapter_path, exist_ok=True)
        model.save_pretrained(init_adapter_path)
        time.sleep(2)
        assert os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")) 

    cot_name = "cot" if args.with_cot else "direct"
    for filename, fulldata in data_list:
        filename = filename.split('.')[0] 
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(
            ROOT_DIR, 
            "offline_prag", 
            args.model_name, 
            args.dataset,
            f"epoch={args.num_train_epochs}_lr={args.learning_rate}",
            filename,
        )
        os.makedirs(output_dir, exist_ok=True)
        fulldata = fulldata if args.sample == -1 else fulldata[:args.sample]
        for did, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            passages = data["passages"]
            data["augment"] = []
            for passage in passages:
                gid = passage["global_id"]
                if gid in aug_map:
                    data["augment"].append(
                        {
                            "global_id": gid,
                            "passage": passage["passage"],
                            "augment": aug_map[gid]
                        }
                    )
            # print(data)
            for pid in range(len(data["augment"])):
                save_path = os.path.join(output_dir, f"data_{did}", f"passage_{pid}")
                if os.path.exists(os.path.join(save_path, "adapter_model.safetensors")):
                    continue
                query = data.get("question", data.get("input"))
                aug_list = data["augment"][pid]
                # print(data["augment"][pid]["augment"])
                model = train(query, aug_list, args, model, tokenizer, 
                            init_adapter_path, save_path)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task_type", type=str, default="open_domain_qa", choices=["open_domain_qa", "fact_checking", "slot_filling"])
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No config for LoRA"
    print(args)
    main(args)