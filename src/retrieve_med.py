import os
import json
import argparse
from tqdm import tqdm

from root_dir_path import ROOT_DIR
from retrieve.retriever import bm25_retrieve_med

import re

def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_yes_no(answer_text: str) -> str:
    text = answer_text.lower()
    if "final decision is:" in text:
        tail = text.split("final decision is: ", 1)[1].strip()
        if tail.startswith("yes"):
            return "yes"
        if tail.startswith("no"):
            return "no"


def load_pubmedqa(data_path):
    pubmedqa_path = os.path.join(data_path, "train.jsonl")
    dataset = []

    with open(pubmedqa_path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            item = json.loads(line.strip())
            answer = extract_yes_no(item["answer"])

            dataset.append({
                "test_id": idx,
                "question": item["question"],
                "answer": answer,
            })

    dataset = dataset[1000:]
    return {"total": dataset}


def main(args):
    output_dir = os.path.join(ROOT_DIR, "data_ret_pub10", "pubmedqa")
    docs_file = os.path.join(ROOT_DIR, "all_docs_med10.json")
    os.makedirs(output_dir, exist_ok=True)

    print("### Loading MedQA ###")
    load_dataset = load_pubmedqa(args.data_path)
    solve_dataset = load_dataset  # only total

    for filename, dataset in solve_dataset.items():
        # load existing docs (avoid duplicates)
        if os.path.exists(docs_file):
            with open(docs_file, "r") as fin:
                all_docs_list = json.load(fin)
            existing_ids = {d["global_id"] for d in all_docs_list}
            print(f"Loaded {len(all_docs_list)} existing docs from {docs_file}")
        else:
            all_docs_list = []
            existing_ids = set()

        output_file = os.path.join(
            output_dir, 
            "total.json"
        )

        done_data = []
        done_ids = set()
        if os.path.exists(output_file):
            with open(output_file, "r") as fin:
                try:
                    done_data = json.load(fin)
                    done_ids = {d["test_id"] for d in done_data}
                    print(f"Found {len(done_ids)} finished samples, skipping them.")
                except:
                    print("Warning: broken output file, restart from scratch")
                    done_data, done_ids = [], set()

        ret = done_data
        dataset = dataset[:args.sample] if args.sample > 0 else dataset

        pbar = tqdm(total=len(dataset) * args.topk)
        cut_id = len(done_ids) * args.topk
        start_with = 0

        for data in dataset:
            if start_with * args.topk < cut_id:
                start_with += 1
                pbar.update(args.topk)
                continue

            q_text = data["question"]
            pids, passages = bm25_retrieve_med(q_text, topk=args.topk + 10)

            final_passages = []
            seen = set()   
            for pid, psg in zip(pids, passages):
                norm_psg = normalize_text(psg)
                if norm_psg in seen:
                    pbar.update(1)
                    continue
                seen.add(norm_psg)
                entry = {
                    "global_id": pid,
                    "passage": psg
                }
                final_passages.append(entry)

                if pid is not None:
                    if pid not in existing_ids:
                        all_docs_list.append({"global_id": pid, "text": psg})
                        existing_ids.add(pid)

                pbar.update(1)
                if len(final_passages) == args.topk:
                    break

            data["passages"] = final_passages
            ret.append(data)

            with open(output_file, "w") as fout:
                json.dump(ret, fout, indent=4)

        with open(docs_file, "w") as fout:
            json.dump(all_docs_list, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    print(args)
    main(args)
