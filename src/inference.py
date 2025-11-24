# this script includes inference methods for different settings: LLM direct, RAG, PRAG, task_lora-only, D-PRAG
# LLM_direct: directly use the base LLM for inference
# RAG: use baseline RAG
# PRAG: use baseline PRAG
# task_lora-only: finetune LoRA with input-output for specific task, this method is the fundation for our method, it should perform better than LLM_direct
# D-PRAG: this is our method: load both task LoRA weak(LoRA trained with input-output for specific task) and document LoRA, merge them for inference
# test: it's something we used for debugging
# Goal: perform better than baseline PRAG and RAG with our D-PRAG method
import os
import gc
import json
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, load_data, read_complete, predict, predict_qa_llm, predict_fc, predict_fc_llm, predict_sf, predict_sf_llm

def main(args):
    if args.dataset in ["fever", "zeroshot_re", "triviaqa"]:
        data_dir = os.path.join(ROOT_DIR, "data_ret_kilt", args.dataset)
    elif args.dataset == "test":
        data_dir = os.path.join(ROOT_DIR, "data_ret_test", args.dataset)
    else:
        data_dir = os.path.join(ROOT_DIR, "data_ret_dpr", args.dataset)
    data_list = load_data(None, None, None, data_dir=data_dir)
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens = args.max_new_tokens,
    )
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)
    
    doc_LoRA_path = os.path.join(
        ROOT_DIR,
        "offline_doc",
        args.model_name,
        args.dataset
    )
    PRAG_LoRA_path = os.path.join(
        ROOT_DIR,
        "offline_prag",
        args.model_name,
        args.dataset,
        f"epoch={args.num_train_epochs}_lr={args.learning_rate}"
    )
    task_LoRA_path = os.path.join(
        ROOT_DIR,
        "offline_task",
        args.model_name,
        args.task_type,
        "LLM"
    )
    output_root_dir = os.path.join(
        ROOT_DIR, 
        "output",
        args.model_name, 
        args.task_type,
        args.dataset,
        args.inference_method
    )
    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(output_root_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)

        predict_file = os.path.join(output_dir, "predict.json")
        ret, start_with = read_complete(predict_file)

        fulldata = fulldata[start_with:] if args.sample == -1 else fulldata[start_with:args.sample]
        # fulldata = [fulldata[27]]
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data.get("question") or data.get("input")
            # print(f"Processing {test_id}: {question}")
            passages = data.get("passages")
            answer = data.get("answer") or data.get("output")
            # print(f"Ground Truth: {answer}")


            def get_pred(model, psgs):
                if args.task_type == "fact_checking":
                    if args.inference_method in ["LLM_direct", "task_lora-only", "D-PRAG"]:
                        text = predict_fc_llm(model, tokenizer, generation_config, 
                                        question)
                    else:
                        text = predict_fc(model, tokenizer, generation_config, 
                                        question, psgs)
                elif args.task_type == "slot_filling":
                    template_question = data["template_question"]
                    # print(f"Template Question: {template_question}")
                    if args.inference_method in ["LLM_direct", "task_lora-only", "D-PRAG"]:
                        text = predict_sf_llm(model, tokenizer, generation_config, 
                                        question, template_question)
                    else:
                        text = predict_sf(model, tokenizer, generation_config, 
                                        question, template_question, psgs)
                else:   # open_domain_qa
                    if args.inference_method in ["LLM_direct", "task_lora-only", "D-PRAG"]:
                        text = predict_qa_llm(model, tokenizer, generation_config, 
                                        question, with_cot=args.with_cot)
                    else:
                        text = predict(model, tokenizer, generation_config, 
                                        question, with_cot=args.with_cot, passages=psgs)
                pred = {
                    "test_id": test_id, 
                    "question": question, 
                    "answer": answer, 
                    "text": text,
                }
                pred.update(evaluate(text, answer, args.with_cot))
                return pred

            if args.inference_method == "LLM_direct":
                ret.append(get_pred(model, psgs=None))
            elif args.inference_method == "RAG":
                ret.append(get_pred(model, psgs=passages))
            elif args.inference_method == "PRAG":
                adapter_names = []
                for pid in range(len(passages)):
                    adapter_path = os.path.join(PRAG_LoRA_path, filename, f"data_{test_id}", f"passage_{pid}")
                    if pid == 0:
                        model = PeftModel.from_pretrained(
                            model, 
                            adapter_path,
                            adapter_name = "0", 
                            is_trainable = False
                        )
                    else:
                        model.load_adapter(adapter_path, adapter_name = str(pid)) 
                    adapter_names.append(str(pid))
                model.add_weighted_adapter(
                    adapters = adapter_names,
                    weights=[1 / len(adapter_names)] * len(adapter_names),
                    adapter_name = "merge",
                    combination_type = "cat",
                )
                model.set_adapter("merge")
                ret.append(get_pred(model, psgs=None))
                model.delete_adapter("merge")
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()
            elif args.inference_method == "task_lora-only":
                model = PeftModel.from_pretrained(
                    model, 
                    task_LoRA_path,
                    adapter_name = "0", 
                    is_trainable = False
                )
                ret.append(get_pred(model, psgs=None))
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()
            elif args.inference_method == "D-PRAG":
                adapter_names = []
                # Load task LoRA
                model = PeftModel.from_pretrained(
                    model, 
                    task_LoRA_path,
                    adapter_name = "0", 
                    is_trainable = False
                )
                adapter_names.append("0")
                # Load document LoRAs
                for pid in range(len(passages)):
                    adapter_path = os.path.join(doc_LoRA_path, filename, "epoch=2_lr=0.0003", f"data_{test_id}", f"passage_{pid}") # TODO: change epoch and lr if needed
                    model.load_adapter(adapter_path, adapter_name = str(pid+1))
                    adapter_names.append(str(pid+1))

                task_weight = args.task_lora_weight
                doc_weight = (1 - task_weight) / (len(adapter_names) - 1)
                weights_list = [task_weight] + [doc_weight] * (len(adapter_names) - 1)
                
                model.add_weighted_adapter(
                    adapters = adapter_names,
                    # weights=[1 / len(adapter_names)] * len(adapter_names), # TODO: try other weighting methods if needed
                    weights = weights_list,
                    adapter_name = "merge",
                    combination_type = "cat",
                )
                model.set_adapter("merge")
                ret.append(get_pred(model, psgs=None))
                model.delete_adapter("merge")
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()
                
            with open(predict_file, "w") as fout:
                json.dump(ret, fout, indent=4)

        ##### Evaluating #####
        metrics = ["em", "f1", "prec", "recall"]
        ret_str = ""
        for met in metrics:
            acc = sum(float(d[met]) for d in ret) / len(ret)
            acc = round(acc, 4)
            ret_str += f"{met}\t{acc}\n"
        ret_str += "\n" + json.dumps(vars(args), indent=4)
        with open(os.path.join(output_dir, "result.txt"), "w") as fout:
            fout.write(ret_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True, choices=["open_domain_qa", "fact_checking", "slot_filling"])
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--task_lora_weight", type=float, default=0.5)
    parser.add_argument("--inference_method", type=str, default="LLM_direct", choices=["FT_RAG", "FT_LLM", "LLM_direct", "RAG", "PRAG",  "task_lora-only", "D-PRAG"])
    # LoRA
    parser.add_argument("--lora_rank", type=int ,default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    print(args)
    main(args)