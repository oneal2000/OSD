import os
import json
import random
import pandas as pd
from tqdm import tqdm
import argparse
from root_dir_path import ROOT_DIR
from retrieve.retriever import bm25_retrieve_kilt

random.seed(42)

# Fact Checking
def load_fever(data_path):
    fever_path = os.path.join(data_path, "fever-dev-kilt.jsonl")
    
    new_dataset = []

    with open(fever_path, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line.strip())
            outputs = data.get("output", [])
            
            answers = [o.get("answer") for o in outputs if "answer" in o]
            unique_answers = set(answers)
            
            if len(unique_answers) == 1:
                val = {
                    "id": data.get("id"),
                    "input": data.get("input"),
                    "answer": unique_answers.pop()
                }
                new_dataset.append(val)

    new_dataset = new_dataset[1000:]
    return {"total": new_dataset}


# Slot filling
def load_zeroshot_re(data_path):
    zsre_path = os.path.join(data_path, "structured_zeroshot-dev-kilt.jsonl")

    new_dataset = []

    with open(zsre_path, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line.strip())
            sample_id = data.get("id")
            sample_input = data.get("input")
            
            outputs = data.get("output", [])
            all_answers = []
            for o in outputs:
                if "answer" in o:
                    if isinstance(o["answer"], list):
                        all_answers.extend(o["answer"])
                    else:
                        all_answers.append(o["answer"])
            
            template_question = ""
            meta = data.get("meta", {})
            template_questions = meta.get("template_questions", [])
            if template_questions:
                template_question = template_questions[0]
            
            val = {
                "id": sample_id,
                "input": sample_input,
                "answer": all_answers,
                "template_question": template_question
            }
            new_dataset.append(val)

    new_dataset = new_dataset[1000:]
    return {"total": new_dataset}

# Open domain QA
def load_triviaqa(data_path):
    triviaqa_path = os.path.join(data_path, "triviaqa-dev-kilt.jsonl")
    new_dataset = []

    with open(triviaqa_path, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line.strip())
            
            sample_id = data.get("id")
            sample_input = data.get("input")
            
            outputs = data.get("output", [])
            all_answers = []
            for o in outputs:
                ans = o.get("answer")
                if ans:
                    if isinstance(ans, list):
                        all_answers.extend(ans)
                    else:
                        all_answers.append(ans)
            
            val = {
                "id": sample_id,
                "input": sample_input,
                "answer": all_answers
            }
            new_dataset.append(val)
    
    new_dataset = new_dataset[1000:]
    return {"total": new_dataset}

# Dialogue
def load_wow(data_path):
    wow_path = os.path.join(data_path, "wow-dev-kilt.jsonl")
    new_dataset = []

    with open(wow_path, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line.strip())
            
            sample_id = data.get("id")
            sample_input = data.get("input")
            
            sample_output = data.get("output", [])
            all_answers = [o.get("answer", "") for o in sample_output if "answer" in o]
            
            val = {
                "id": sample_id,
                "input": sample_input,
                "answer": all_answers
            }
            new_dataset.append(val)
    
    return {"total": new_dataset}


def main(args):
    output_dir = os.path.join(ROOT_DIR, "FT_data", args.dataset)
    # docs_file = os.path.join(ROOT_DIR, "all_docs_kilt.json")
    os.makedirs(output_dir, exist_ok=True)

    print("### Loading dataset ###")
    if f"load_{args.dataset}" in globals():
        load_func = globals()[f"load_{args.dataset}"]
    else:
        load_func = globals()["load_default_format_data"]
    load_dataset = load_func(args.data_path)
    if len(load_dataset) == 1:
        solve_dataset = load_dataset
    else:
        solve_dataset = {k: v for k, v in load_dataset.items() if k != "total"}
        with open(os.path.join(output_dir, "total.json"), "w") as fout:
            json.dump(load_dataset["total"][:args.sample], fout, indent=4)

    for filename, dataset in solve_dataset.items():
        # if os.path.exists(docs_file):
        #     with open(docs_file, "r") as fin:
        #         all_docs_list = json.load(fin)
        #     existing_ids = {d["global_id"] for d in all_docs_list}
        #     print(f"Loaded {len(all_docs_list)} existing docs from {docs_file}")
        # else:
        #     all_docs_list = []
        #     existing_ids = set()
            
        print(f"### Solving {filename} ###")
        output_file = os.path.join(
            output_dir, 
            filename if filename.endswith(".json") else filename + ".json"
        )

        done_data = []
        done_ids = set()

        if os.path.exists(output_file):
            with open(output_file, "r") as fin:
                try:
                    done_data = json.load(fin)
                    done_ids = {d["test_id"] for d in done_data}
                    print(f"Found {len(done_ids)} finished samples, skipping them.")
                except Exception as e:
                    print("Warning: output file broken, restart from scratch:", e)
                    done_data = []
                    done_ids = set()

        ret = done_data
        dataset = dataset[:args.sample]
        pbar = tqdm(total = args.sample * args.topk)
        cut_id = len(done_ids) * args.topk
        start_with = 0
        for data in dataset:
            if start_with * args.topk < cut_id:
                start_with += 1
                pbar.update(args.topk)
                continue
            passage_ids, passages = bm25_retrieve_kilt(data["input"], topk=args.topk+10)
            final_passages = []
            for pid, psg in zip(passage_ids, passages):
                val = { 
                    "global_id": pid, 
                    "passage": psg, 
                }
                final_passages.append(val)
                # if pid not in existing_ids:
                #     all_docs_list.append({"global_id": pid, "text": psg})
                #     existing_ids.add(pid)
                pbar.update(1)
                if len(final_passages) == args.topk:
                    break
            data["passages"] = final_passages
            ret.append(data)

            with open(output_file, "w") as fout:
                json.dump(ret, fout, indent=4)

        # with open(docs_file, "w") as fout:
        #     json.dump(all_docs_list, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="/data-share/yeesuanAI08/zhanghanwen/D-PRAG/data_kilt")
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--topk", type=int, default=3) 
    args = parser.parse_args()
    print(args)
    main(args)