import os
import gc
import sys
import time
import json
import argparse
import torch
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig, PeftModel
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from typing import Dict, List
import prompt_template
from prompt_template import *
from root_dir_path import ROOT_DIR
from utils import get_model, load_data
import numpy as np
import random
from safetensors.torch import load_file

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class TrainingData(Dataset):
    ignored_id = -100

    def __init__(self, prompt_ids, tokenizer, args):
        max_length = args.block_size
        self.dataset = []
        self.max_raw_len = 0
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for input_ids in prompt_ids:
            raw_len = len(input_ids)
            if raw_len > self.max_raw_len:
                self.max_raw_len = raw_len
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
        # print(f"Processed {self.total_len} samples. Max sequence length: {self.max_raw_len}")

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

    qas, fcs, sfs, dias, pubmedqas= [], [], [], [], []
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
    elif args.task_type == "dialogue":
        dias = augments["dialogue"]
        qpa_cnt = (len(dias) + 1) // 2
    elif args.task_type == "med_verify":
        pubmedqas = augments["pubmedqa"]
        qpa_cnt = (len(pubmedqas) + 1) // 2
    
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
                        get_prompt(
                            tokenizer,
                            qa["question"],
                            None,
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
                prompt_ids.append(get_prompt_fc(tokenizer, fc["input"], None, fc["output"]))

    elif args.task_type == "slot_filling":
        for sid, sf in enumerate(sfs):
            if sid < qpa_cnt:
                for ppp in [psg, rew]:
                    prompt_ids.append(get_prompt_sf(tokenizer, sf["input"], sf["template_question"], [ppp], sf["output"]))
            else:
                prompt_ids.append(get_prompt_sf(tokenizer, sf["input"], sf["template_question"], None, sf["output"]))

    elif args.task_type == "med_verify":
        for pid, qa in enumerate(pubmedqas):
            if pid < qpa_cnt:
                for ppp in [psg, rew]:
                    prompt_ids.append(
                        get_prompt(
                            tokenizer,
                            qa["question"],
                            [ppp],
                            qa["answer"]
                        )
                    )
            else:
                prompt_ids.append(
                    get_prompt(
                        tokenizer,
                        qa["question"],
                        None,
                        qa["answer"]
                    )
                )

    elif args.task_type == "dialogue":
        for did, dia in enumerate(dias):
            if did < qpa_cnt:
                for ppp in [psg, rew]:
                    prompt_ids.append(get_prompt_dialogue(tokenizer, dia["input"], [ppp], dia["output"]))
            else:
                prompt_ids.append(get_prompt_dialogue(tokenizer, dia["input"], None, dia["output"]))
                        

    return prompt_ids

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

def _normalize_module_name(name: str) -> str:
    if "model.layers" in name:
        idx = name.index("model.layers")
        return name[idx:]
    return name


def load_task_lora_weights(task_lora_path: str) -> Dict[str, torch.Tensor]:
    weight_path = os.path.join(task_lora_path, "adapter_model.safetensors")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Cannot find task LoRA weights at {weight_path}")
    state_dict = load_file(weight_path)
    # print(state_dict)
    allowed_suffix = {"down_proj", "gate_proj", "up_proj"}
    task_params = {}
    for key, tensor in state_dict.items():
        if key.endswith("lora_A.weight"):
            module_name = key.rsplit(".lora_A.weight", 1)[0]
            suffix = module_name.rsplit('.', 1)[-1]
            if suffix in allowed_suffix:
                norm_name = _normalize_module_name(module_name)
                task_params[norm_name] = tensor.clone()
    return task_params

def orthogonal_loss(model, task_lora_params):
    device = next(model.parameters()).device
    loss = torch.tensor(0.0, device=device)

    if not task_lora_params:
        return loss

    doc_params = {}
    for name, param in model.named_parameters():
        if f".lora_A.default.weight" in name:
            module_name = name.split(f".lora_A.default.weight")[0]
            suffix = module_name.rsplit('.', 1)[-1]
            if suffix in {"down_proj", "gate_proj", "up_proj"}:
                # print(name)
                norm_name = _normalize_module_name(module_name)
                doc_params[norm_name] = param.to(device)

    # for name, param in doc_params.items():
    #     if any(s in name for s in ["down_proj", "gate_proj", "up_proj"]):
    #         print(param)
    #         print(f"[D] {name} sum={param.abs().sum().item():.6f}")

    normalized_task = {k: v.to(device) for k, v in task_lora_params.items()}
    # for name, param in normalized_task.items():
    #     if any(s in name for s in ["down_proj", "gate_proj", "up_proj"]):
    #         print(param)
    #         print(f"[T] {name} sum={param.abs().sum().item():.6f}")
    
    common_modules = set(normalized_task.keys()) & set(doc_params.keys())
    # print(common_modules)

    for module_name in common_modules:
        task_param = normalized_task[module_name]
        doc_param = doc_params[module_name]
        doc_flat = doc_param.view(doc_param.size(0), -1)
        task_flat = task_param.view(task_param.size(0), -1)
        assert doc_flat.shape == task_flat.shape, f"Shape mismatch: {doc_flat.shape} vs {task_flat.shape}"
        current_loss = torch.trace(task_flat @ doc_flat.T @ doc_flat @ task_flat.T)
        loss = loss + current_loss

    return loss

def _global_grad_norm_from_grads(grads) -> float:
    sq_sum = 0.0
    for g in grads:
        if g is None:
            continue
        n = g.detach().norm(2).item()
        sq_sum += n * n
    return float(sq_sum ** 0.5)


def train(model, augments,  tokenizer, args, 
          init_adapter_path, save_path, task_lora_params):
    prompt_ids = get_train_data(augments, tokenizer, args)
    train_data = TrainingData(prompt_ids, tokenizer, args)
    model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
    model.is_parallelizable = True
    model.model_parallel = True
    device = next(model.parameters()).device
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, device),
        shuffle=False,
    )
    task_lora_params = {k: v.to(device) for k, v in task_lora_params.items()}
    # for name, param in task_lora_params.items():
    #     if any(s in name for s in ["down_proj", "gate_proj", "up_proj"]):
    #         print(f"{name} sum={param.abs().sum().item():.6f}")
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)

    first_out_loss = None
    first_ortho_loss = None

    out_loss_hist = []
    ortho_loss_hist = []

    for epoch in range(args.num_train_epochs):
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(loop):
            optimizer.zero_grad()
            outputs = model(**batch)
            
            out_loss = outputs.loss
            ortho = orthogonal_loss(model, task_lora_params)
            loss = out_loss + args.lambda_orth * ortho
            # loss = out_loss

            if first_out_loss is None and first_ortho_loss is None:
                first_out_loss = out_loss.item()
                first_ortho_loss = ortho.item()

            out_loss_hist.append(out_loss.item())
            ortho_loss_hist.append(ortho.item())

            loss.backward()
            optimizer.step()

            loop.set_postfix({
                "out_loss": f"{out_loss.item():.4f}",
                "ortho_loss": f"{ortho.item():.4f}",
                "total_loss": f"{loss.item():.4f}"
            })
    # trainable_params = [p for p in model.parameters() if p.requires_grad]

    # for epoch in range(args.num_train_epochs):
    #     loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    #     for step, batch in enumerate(loop):
    #         optimizer.zero_grad()
    #         outputs = model(**batch)

    #         out_loss = outputs.loss
    #         ortho = orthogonal_loss(model, task_lora_params)

    #         if epoch == 0 and step == 0:
    #             grads_out = torch.autograd.grad(
    #                 out_loss, trainable_params, retain_graph=True, allow_unused=True
    #             )
    #             norm_out = _global_grad_norm_from_grads(grads_out)

    #             grads_ortho = torch.autograd.grad(
    #                 ortho, trainable_params, retain_graph=True, allow_unused=True
    #             )
    #             norm_ortho = _global_grad_norm_from_grads(grads_ortho)

    #             print(f"\n" + "="*50)
    #             print(f"[Gradient Probe]")
    #             print(f"CE Loss:    {out_loss.item():.6f}  =>  CE Grad Norm:    {norm_out:.6f}")
    #             print(f"Ortho Loss: {ortho.item():.6f}  =>  Ortho Grad Norm: {norm_ortho:.6f}")

    #             suggested_ratio = norm_out / (norm_ortho + 1e-8)
    #             print(f" 1:1  (CE_Norm / Ortho_Norm): {suggested_ratio:.6f}")
    #             print(f"current lambda_orth: {args.lambda_orth}")
    #             print(f"lambda_orth (10%~20%): {suggested_ratio * 0.15:.6f}")
    #             print("="*50 + "\n")

    #         loss = out_loss + args.lambda_orth * ortho
    #         # loss = out_loss

    #         if first_out_loss is None and first_ortho_loss is None:
    #             first_out_loss = out_loss.item()
    #             first_ortho_loss = ortho.item()

    #         out_loss_hist.append(out_loss.item())
    #         ortho_loss_hist.append(ortho.item())

    #         loss.backward()
    #         optimizer.step()

    #         loop.set_postfix({
    #             "out_loss": f"{out_loss.item():.4f}",
    #             "ortho_loss": f"{ortho.item():.4f}",
    #             "total_loss": f"{loss.item():.4f}"
    #         })


    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    model = model.unload()
    torch.cuda.empty_cache()
    gc.collect()
    return model,first_out_loss,first_ortho_loss,out_loss_hist,ortho_loss_hist


def main(args):
    if args.dataset == "fever":
        data_dir = os.path.join(ROOT_DIR, "data_ret_kilt10", args.dataset)
        kilt_aug_file = os.path.join(ROOT_DIR, "doc_aug", "kilt.json")
        if os.path.exists(kilt_aug_file):
            aug_file = kilt_aug_file
        else:
            aug_file = os.path.join(ROOT_DIR, "doc_aug", "fever.json")
    elif args.dataset == "zeroshot_re":
        data_dir = os.path.join(ROOT_DIR, "data_ret_kilt10", args.dataset)
        kilt_aug_file = os.path.join(ROOT_DIR, "doc_aug", "kilt.json")
        if os.path.exists(kilt_aug_file):
            aug_file = kilt_aug_file
        else:
            aug_file = os.path.join(ROOT_DIR, "doc_aug", "zsre.json")
    elif args.dataset == "wow":
        data_dir = os.path.join(ROOT_DIR, "data_ret_kilt10", args.dataset)
        kilt_aug_file = os.path.join(ROOT_DIR, "doc_aug", "kilt.json")
        if os.path.exists(kilt_aug_file):
            aug_file = kilt_aug_file
        else:
            aug_file = os.path.join(ROOT_DIR, "doc_aug", "wow.json")
    elif args.dataset == "pubmedqa":
        data_dir = os.path.join(ROOT_DIR, "data_ret_pub10", args.dataset)
        aug_file = os.path.join(ROOT_DIR, "doc_aug", "med.json") 
    else:
        data_dir = os.path.join(ROOT_DIR, "data_ret_dpr10", args.dataset)
        aug_file = os.path.join(ROOT_DIR, "doc_aug", "dpr.json")
    data_list = load_data(None, None, data_type="total", data_dir=data_dir)

    with open(aug_file, "r", encoding="utf-8") as f:
        aug_data_list = json.load(f)

    aug_map = {item["global_id"]: item["augment"] for item in aug_data_list}

    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)
    
    task_lora_path = os.path.join(
            ROOT_DIR,
            "offline_task",
            args.model_name,
            args.task_type
        )

    task_base_path = os.path.join(
        ROOT_DIR,
        "task_base_LLM",
        args.model_name,
        args.task_type
    )

    task_lora_cache = {}

    for filename, fulldata in data_list:
        filename = filename.split('.')[0] 
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(
            ROOT_DIR, 
            "offline_doc", 
            args.model_name, 
            f"lambda={args.lambda_orth}",
            args.dataset,
            filename,
            f"epoch={args.num_train_epochs}_lr={args.learning_rate}",
        )

        base_model, tokenizer, _ = get_model(args.model_name)
        task_path_current = task_lora_path
        task_base_save_path = os.path.join(task_base_path)
        model = load_task_lora_as_base(base_model, task_lora_path, task_base_save_path, tokenizer)
        model, tokenizer, _ = get_model(task_base_save_path)

        cache_key = task_path_current
        print(f"Loading task LoRA from {cache_key}")
        if cache_key not in task_lora_cache:
            task_lora_cache[cache_key] = load_task_lora_weights(cache_key)
        task_lora_params = task_lora_cache[cache_key]
        # print(task_lora_params)

        init_path = os.path.join(
            ROOT_DIR, 
            "offline_doc", 
            args.model_name, 
            "base_weight",
            args.task_type
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

            del model
            torch.cuda.empty_cache()
            gc.collect()
            model, tokenizer, _ = get_model(task_base_save_path)

        os.makedirs(output_dir, exist_ok=True)
        fulldata = fulldata if args.sample == -1 else fulldata[:args.sample]

        first_out_losses = []
        first_ortho_losses = []
        all_loss = []

        for did, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            task_field_map = {
                "open_domain_qa": "qa",
                "fact_checking": "fact_checking",
                "slot_filling": "slot_filling",
                "dialogue": "dialogue",
                "med_verify" : "pubmedqa"
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

            data_loss_records = {}

            # print(data)
            for pid in range(len(data["augment"])):
                passage_id = pid
                save_path = os.path.join(output_dir, f"data_{did}", f"passage_{passage_id}")
                check_path = os.path.join(save_path)
                if os.path.exists(os.path.join(check_path, "adapter_model.safetensors")):
                    continue
                aug_list = data["augment"][pid]
                # print(data["augment"][pid])
                model,first_out,first_ortho,out_hist,ortho_hist = train(model, aug_list, tokenizer, args, init_path, save_path, task_lora_params)
                if first_out is not None and first_ortho is not None:
                    first_out_losses.append(first_out)
                    first_ortho_losses.append(first_ortho)

                data_loss_records[pid] = {
                    "out_loss": out_hist,
                    "ortho_loss": ortho_hist,
                }

            if len(data_loss_records) > 0:
                for pid in sorted(data_loss_records.keys()):
                        out_hist = data_loss_records[pid]["out_loss"]
                        ortho_hist = data_loss_records[pid]["ortho_loss"]
                        out_str = ", ".join(f"{v:.6f}" for v in out_hist)
                        ortho_str = ", ".join(f"{v:.6f}" for v in ortho_hist)
                        all_loss.append(f"data{did}:passage{pid} outloss：{out_str} ortholoss：{ortho_str}")
                
        if len(first_ortho_losses) > 0:
            avg_out = sum(first_out_losses) / len(first_out_losses)
            avg_ortho = sum(first_ortho_losses) / len(first_ortho_losses)
            loss_file = os.path.join(output_dir, "loss_compare.txt")
            with open(loss_file, "w", encoding="utf-8") as f:
                f.write(f"avg_first_out_loss={avg_out:.6f}\n")
                f.write(f"avg_first_ortho_loss={avg_ortho:.6f}\n")
        
        if len(all_loss) >0:
            all_detail_path = os.path.join(output_dir, "loss_detail.txt")
            with open(all_detail_path, "w", encoding="utf-8") as f:
                f.write("\n".join(all_loss) + "\n")


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
    parser.add_argument("--lambda_orth", type=float, default=10.0)
    parser.add_argument("--block_size", type=int, default=500)
    args = parser.parse_args()
    print(args)
    main(args)