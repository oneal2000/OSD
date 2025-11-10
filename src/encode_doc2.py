import os
import gc
import sys
import time
from typing import Optional, Any
import json
import argparse
import torch
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig, PeftModel, PeftMixedModel
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from typing import Dict, List
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


def train(model, data, tokenizer, args, save_path, task_lora_path, doc_lora_init_path):
    prompt_ids = get_train_data(data, tokenizer, args)

    train_data = TrainingData(prompt_ids, tokenizer, args)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=False
    )

    model = PeftModel.from_pretrained(model, task_lora_path, adapter_name="0", is_trainable=False)
    model.load_adapter(doc_lora_init_path, adapter_name="1", is_trainable=True)
    
    for name, param in model.named_parameters():
        if ".0." in name and "lora" in name:
            param.requires_grad = False
        elif ".1." in name and "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False 
    
    # trainable_params = [(name, param.numel()) for name, param in model.named_parameters() if param.requires_grad]
    # print(f"Trainable parameters: {sum(p[1] for p in trainable_params)}")
    # for name, count in trainable_params:
    #     if "lora" in name:
    #         print(f"  {name}: {count}")
    
    original_forward = model.forward
    
    def combined_forward(**kwargs):
        model.set_adapter("0")
        with torch.no_grad():
            task_output = original_forward(**kwargs)
        
        model.set_adapter("1")
        doc_output = original_forward(**kwargs)
        
        combined_logits = task_output.logits * 0.2 + doc_output.logits * 0.8
        
        return type(doc_output)(
            loss=doc_output.loss,
            logits=combined_logits,
            hidden_states=doc_output.hidden_states,
            attentions=doc_output.attentions
        )
    
    model.forward = combined_forward

    model.is_parallelizable = True
    model.model_parallel = True
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)

    for epoch in range(args.num_train_epochs):
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            optimizer.zero_grad()

            # pre_params = {}
            # for name, param in model.named_parameters():
            #     if ("lora" in name) and (".0." in name or ".1." in name):
            #         pre_params[name] = param.detach().clone()

            outputs = model(**batch)
            ce_loss = outputs.loss

            ortho_loss = orthogonal_loss(model, doc_adapter_name="1", task_adapter_name="0")

            loss = ce_loss + args.lambda_orth * ortho_loss
            loss.backward()
            optimizer.step()

            # for name, param in model.named_parameters():
            #     if name in pre_params:
            #         diff = (param - pre_params[name]).abs().sum().item()
            #         print(f"{name}, change: {diff:.6f}")

            loop.set_postfix({
                "ce_loss": f"{ce_loss.item():.4f}",
                "ortho_loss": f"{ortho_loss.item():.4f}",
                "total_loss": f"{loss.item():.4f}"
            })

    model.forward = original_forward
    
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
        # data_dir = os.path.join(ROOT_DIR, "data_ret_test", "test")
        # aug_file = os.path.join(ROOT_DIR, "ex.json")
    # elif args.dataset == "test":
    #     data_dir = os.path.join(ROOT_DIR, "data_ret_test", "test")
    #     aug_file = os.path.join(ROOT_DIR, "doc_aug", "dpr_test.json")
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

    model, tokenizer, _ = get_model(args.model_name)
    
    if args.task_LoRA_type == "strong":
        task_lora_path = os.path.join(
            ROOT_DIR,
            "offline_FT",
            args.model_name,
            args.dataset,
            "LLM",
            "batch=8_epoch=1_lr=0.0001_dropout=0.2"
        )

    doc_lora_init_path = os.path.join(
        ROOT_DIR,
        "offline_doc",
        args.model_name,
        "base_weight"
    )
    if not os.path.exists(os.path.join(doc_lora_init_path, "adapter_model.safetensors")):
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
        print(f'Save LoRA base weight to {doc_lora_init_path}')
        os.makedirs(doc_lora_init_path, exist_ok=True)
        model.save_pretrained(doc_lora_init_path)
        time.sleep(2)
        assert os.path.exists(os.path.join(doc_lora_init_path, "adapter_model.safetensors")) 

    for filename, fulldata in data_list:
        filename = filename.split('.')[0] 
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(
            ROOT_DIR, 
            "offline_doc", 
            args.model_name, 
            args.dataset,
            f"epoch={args.num_train_epochs}_lr={args.learning_rate}",
            filename,
        )
        task_path = os.path.join(task_lora_path, filename)
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
                if os.path.exists(os.path.join(save_path, "adapter_model.safetensors")):
                    continue
                aug_list = data["augment"][pid]
                # print(data["augment"][pid])
                model = train(model, aug_list, tokenizer, args, save_path, task_path, doc_lora_init_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lambda_orth", type=float, default=0.1)
    parser.add_argument("--task_LoRA_type", type=str, choices=["strong", "weak"], default="strong")
    parser.add_argument("--block_size", type=int, default=1500)
    args = parser.parse_args()
    print(args)
    main(args)