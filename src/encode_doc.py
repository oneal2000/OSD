# this script is to train document-specific LoRA adapters with orthogonal regularization
# first load the task LoRA adapter and merge it into the base model to get the task base model and save it
# then train document LoRA adapters on augmented data with orthogonal regularization against the task LoRA adapters
# the training process is based on the base model with the task base model
import os
import gc
import sys
import time
from typing import Optional, Any
import json
import argparse
import torch
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig, PeftModel, PeftMixedModel, PeftMixedModel
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from typing import Dict, List
import prompt_template
from prompt_template import get_prompt, get_prompt_llm, get_prompt_fc, get_prompt_fc_llm, get_prompt_sf, get_prompt_sf_llm
from root_dir_path import ROOT_DIR
from utils import get_model, load_data
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class TrainingData(Dataset):
    ignored_id = -100

    def __init__(self, prompt_ids, tokenizer, args):
        max_length = args.block_size
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

    def __len__(self):
        return len(self.dataset)

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

    qas, fcs, sfs = [], [], []
    qpa_cnt = 0

    rew = augments["rewrite"]
    if args.task_type == "open_domain_qa":
        qas = augments["qa"]
        qpa_cnt = (len(qas) + 1) // 2
    elif args.task_type == "fact_checking":
        fcs = augments["fact_checking"]
        qpa_cnt = (len(fcs) + 1) // 2
    elif args.task_type == "slot_filling":
        sfs = augments["slot_filling"]
        qpa_cnt = (len(sfs) + 1) // 2
    

    if args.task_type == "open_domain_qa":
        for qid, qa in enumerate(qas):
            if qid < qpa_cnt:
                for ppp in [psg, rew]:
                    prompt_ids.append(
                        get_prompt(
                            tokenizer,
                            qa["question"],
                            [ppp],
                            qa["answer"] if not args.with_cot else qa["full_answer"],
                            with_cot=args.with_cot,
                        )
                    )
            else:
                prompt_ids.append(
                    get_prompt_llm(
                        tokenizer,
                        qa["question"],
                        qa["answer"] if not args.with_cot else qa["full_answer"],
                        with_cot=args.with_cot,
                    )
                )

    elif args.task_type == "fact_checking":
        for fid, fc in enumerate(fcs):
            if fid < qpa_cnt:
                for ppp in [psg, rew]:
                    prompt_ids.append(get_prompt_fc(tokenizer, fc["input"], [ppp], fc["output"]))
            else:
                prompt_ids.append(get_prompt_fc_llm(tokenizer, fc["input"], fc["output"]))

    elif args.task_type == "slot_filling":
        for sid, sf in enumerate(sfs):
            if sid < qpa_cnt:
                for ppp in [psg, rew]:
                    prompt_ids.append(get_prompt_sf(tokenizer, sf["input"], sf["template_question"], [ppp], sf["output"]))
            else:
                prompt_ids.append(get_prompt_sf_llm(tokenizer, sf["input"], sf["template_question"], sf["output"]))

    return prompt_ids

# load the task LoRA adapter, merge it into the base model, and save the new model as the task base model
def load_task_lora_as_base(model, task_lora_path, save_path, tokenizer=None):
    print(f"Loading task LoRA from {task_lora_path}")
    model = PeftModel.from_pretrained(model, task_lora_path)
    model = model.merge_and_unload()
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
    print(f"New task_base LLM saved at {save_path}")
    return model

# orthogonal regularization loss between document LoRA and task LoRA
# this orthogonal loss is computed on the LoRA A matrices
# TODO: test orthogonal loss computed on LoRA B matrices
def orthogonal_loss(model, doc_adapter_name="1", task_adapter_name="0"):
    device = next(model.parameters()).device
    loss = torch.tensor(0.0, device=device)

    task_params = {}
    for name, param in model.named_parameters():
        if f".lora_A.{task_adapter_name}.weight" in name:
            module_name = name.split(f'.lora_A.{task_adapter_name}.weight')[0]
            task_params[module_name] = param.to(device)

    for name, param in model.named_parameters():
        if f".lora_A.{doc_adapter_name}.weight" in name:
            module_name = name.split(f'.lora_A.{doc_adapter_name}.weight')[0]
            if module_name in task_params:
                task_param = task_params[module_name]
                doc_param = param.view(param.size(0), -1).to(device)
                task_param = task_param.view(task_param.size(0), -1)
                loss += torch.norm(doc_param.T @ task_param, p='fro') ** 2

    return loss


def train(model, augments,  tokenizer, args, 
          init_adapter_path, task_path, save_path):
    prompt_ids = get_train_data(augments, tokenizer, args)
    train_data = TrainingData(prompt_ids, tokenizer, args)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=False,
    )
    model = PeftModel.from_pretrained(model, task_path, adapter_name="0", is_trainable=False)
    model.load_adapter(init_adapter_path, adapter_name="1", is_trainable=True)
    model.set_adapter("1")
    model.is_parallelizable = True
    model.model_parallel = True
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)
    for epoch in range(args.num_train_epochs):
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(loop):
            optimizer.zero_grad()
            outputs = model(**batch)
            
            out_loss = outputs.loss
            ortho = orthogonal_loss(model, doc_adapter_name="1", task_adapter_name="0")
            loss = out_loss + args.lambda_orth * ortho
            # loss = out_loss

            loss.backward()
            optimizer.step()

            loop.set_postfix({
                "out_loss": f"{out_loss.item():.4f}",
                "ortho_loss": f"{ortho.item():.4f}",
                "total_loss": f"{loss.item():.4f}"
            })
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, selected_adapters=["1"])
    model.delete_adapter("0")
    model.delete_adapter("1")
    model = model.unload()
    torch.cuda.empty_cache()
    gc.collect()
    return model


def main(args):
    if args.dataset in ["fever", "zeroshot_re", "triviaqa"]:
        data_dir = os.path.join(ROOT_DIR, "data_ret_kilt", args.dataset)
        aug_file = os.path.join(ROOT_DIR, "doc_aug", "kilt_3.json")
    else:
        data_dir = os.path.join(ROOT_DIR, "data_ret_dpr", args.dataset)
        aug_file = os.path.join(ROOT_DIR, "doc_aug", "dpr.json")
    data_list = load_data(None, None, None, data_dir=data_dir)

    with open(aug_file, "r", encoding="utf-8") as f:
        aug_data_list = json.load(f)

    aug_map = {item["global_id"]: item["augment"] for item in aug_data_list}

    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    
    if args.task_LoRA_type == "strong":
        task_lora_path = os.path.join(
            ROOT_DIR,
            "offline_FT",
            args.model_name,
            args.dataset,
            "LLM",
            "batch=8_epoch=1_lr=0.0001_dropout=0.2"
        )
    else:
        task_lora_path = os.path.join(
            ROOT_DIR,
            "offline_task",
            args.model_name,
            args.task_type,
            "LLM"
        )

    task_base_path = os.path.join(
        ROOT_DIR,
        "task_base_LLM",
        args.model_name,
        args.dataset
    )

    task_base_path_weak = os.path.join(
        ROOT_DIR,
        "task_base_LLM_weak",
        args.model_name,
        args.dataset
    )

    for filename, fulldata in data_list:
        filename = filename.split('.')[0] 
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(
            ROOT_DIR, 
            "offline_doc", 
            args.model_name, 
            args.dataset,
            filename,
            f"epoch={args.num_train_epochs}_lr={args.learning_rate}",
        )

        base_model, tokenizer, _ = get_model(args.model_name)
        if args.task_LoRA_type == "strong":
            task_path = os.path.join(task_lora_path, filename)
            task_base_save_path = os.path.join(task_base_path, filename, args.task_LoRA_type)
            model = load_task_lora_as_base(base_model, task_path, task_base_save_path, tokenizer)
            model, tokenizer, _ = get_model(task_base_save_path)
        else:
            task_base_save_path = os.path.join(task_base_path_weak, filename, args.task_LoRA_type)
            model = load_task_lora_as_base(base_model, task_lora_path, task_base_save_path, tokenizer)
            model, tokenizer, _ = get_model(task_base_save_path)

        init_path = os.path.join(
            ROOT_DIR, 
            "offline_doc", 
            args.model_name, 
            args.dataset,
            filename,
            "base_weight"
        )

        if not os.path.exists(os.path.join(init_path, "adapter_model.safetensors")):
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=['down_proj', 'gate_proj', 'up_proj'],
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0,
            )
            model = get_peft_model(model, peft_config)
            model.is_parallelizable = True
            model.model_parallel = True
            print(f'Save LoRA base weight to {init_path}')
            os.makedirs(init_path, exist_ok=True)
            model.save_pretrained(init_path)
            time.sleep(2)
            assert os.path.exists(os.path.join(init_path, "adapter_model.safetensors")) 

        os.makedirs(output_dir, exist_ok=True)
        fulldata = fulldata if args.sample == -1 else fulldata[:args.sample]
        for did, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            task_field_map = {
                "open_domain_qa": "qa",
                "fact_checking": "fact_checking",
                "slot_filling": "slot_filling"
            }

            passages = data["passages"]
            data["augment"] = []
            for passage in passages:
                gid = passage["global_id"]
                if gid in aug_map:
                    # print(aug_map[gid])
                    _to_add = []
                    field_name = task_field_map[args.task_type]
                    _to_add.append({
                        "passage": passage["passage"],
                        "rewrite": aug_map[gid][0]["rewrite"],
                        field_name: aug_map[gid][0][field_name]
                    })
                        
                    data["augment"].extend(_to_add)

            # print(data)
            for pid in range(len(data["augment"])):
                save_path = os.path.join(output_dir, f"data_{did}", f"passage_{pid}")
                check_path = os.path.join(save_path, "1")
                if os.path.exists(os.path.join(check_path, "adapter_model.safetensors")):
                    continue
                aug_list = data["augment"][pid]
                # print(data["augment"][pid])
                if args.task_LoRA_type == "strong":
                    model = train(model, aug_list, tokenizer, args, init_path, task_path, save_path)
                else:
                    model = train(model, aug_list, tokenizer, args, init_path, task_lora_path, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lambda_orth", type=float, default=0.1)
    parser.add_argument("--task_LoRA_type", type=str, choices=["strong", "weak"], default="weak")
    parser.add_argument("--block_size", type=int, default=1500)
    args = parser.parse_args()
    print(args)
    main(args)