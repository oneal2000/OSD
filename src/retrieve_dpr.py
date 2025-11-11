# this script is to retrieve passages for various QA datasets using BM25 retriever in dpr
# TODO: add strategy dataset
import os
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm
from root_dir_path import ROOT_DIR
from retrieve.retriever import bm25_retrieve

random.seed(42)


def load_popqa(data_path):
    data_path = os.path.join(data_path, "popQA.tsv")
    dataset = pd.read_csv(data_path, sep="\t")
    dataset = dataset[1000:]
    new_dataset = []
    for did in range(len(dataset)):
        data = dataset.iloc[did]
        question = data["question"]
        answer = [data["obj"]] + eval(data["o_aliases"])
        val = {
            "test_id": did, 
            "question": question, 
            "answer": answer,
        }        
        new_dataset.append(val)
    return {"total": new_dataset}


def load_complexwebquestions(data_path):
    data_path = os.path.join(data_path, "ComplexWebQuestions_dev.json")
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    dataset = dataset[1000:]
    new_dataset = []
    for did, data in enumerate(dataset):
        question = data["question"]
        answer = []
        for ans in data["answers"]:
            answer.append(ans["answer"])
            answer.extend(ans["aliases"])
        answer = list(set(answer))
        val = {
            "test_id": did, 
            "question": question, 
            "answer": answer,
        }        
        new_dataset.append(val)
    ret = {"total": new_dataset}
    return ret


def load_2wikimultihopqa(data_path):
    with open(os.path.join(data_path, "dev.json"), "r") as fin:
        dataset = json.load(fin)
    mark_idx = {}
    for did, data in enumerate(dataset):
        typ = data["type"]
        if typ not in mark_idx:
            mark_idx[typ] = {"cnt": 1, "last_idx": did}
        else:
            if mark_idx[typ]["cnt"] >= 300:
                continue
            mark_idx[typ]["cnt"] += 1
            mark_idx[typ]["last_idx"] = did
    last_idx = max(v["last_idx"] for k, v in mark_idx.items())
    dataset = dataset[last_idx + 1000:]
    with open(os.path.join(data_path, "id_aliases.json"), "r") as fin:
        aliases = dict()
        for li in fin:
            t = json.loads(li)
            aliases[t["Q_id"]] = t["aliases"]
    new_dataset = []
    type_to_dataset = {}
    for did, data in enumerate(dataset):
        ans_id = data["answer_id"]
        val = {
            "qid": data["_id"], 
            "test_id": did, 
            "question": data["question"], 
            "answer": aliases[ans_id] if ans_id else data["answer"]
        }
        golden_passages = []
        contexts = {name: " ".join(sents) for name, sents in data["context"]}
        for fact_name, _sent_id in data["supporting_facts"]:
            psg = contexts[fact_name]
            golden_passages.append(psg)
        val["golden_passages"] = golden_passages
        val["type"] = data["type"]
        new_dataset.append(val)
        if data["type"] not in type_to_dataset:
            type_to_dataset[data["type"]] = []
        type_to_dataset[data["type"]].append(val)
    ret = {"total": new_dataset}
    ret.update(type_to_dataset)
    return ret


def load_hotpotqa(data_path):
    data_path = os.path.join(data_path, "hotpot_dev_distractor_v1.json")
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    mark_idx = {}
    for did, data in enumerate(dataset):
        typ = data["type"]
        if typ not in mark_idx:
            mark_idx[typ] = {"cnt": 1, "last_idx": did}
        else:
            if mark_idx[typ]["cnt"] >= 300:
                continue
            mark_idx[typ]["cnt"] += 1
            mark_idx[typ]["last_idx"] = did
    last_idx = max(v["last_idx"] for k, v in mark_idx.items())
    dataset = dataset[last_idx + 1000:]
    new_dataset = []
    type_to_dataset = {}
    for did, data in enumerate(dataset):
        val = {
            "qid": data["_id"], 
            "test_id": did, 
            "question": data["question"], 
            "answer": data["answer"]
        }
        tmp = []
        contexts = {name: "".join(sents) for name, sents in data["context"]}
        for fact_name, _sent_id in data["supporting_facts"]:
            psg = contexts[fact_name]
            tmp.append(psg)
        golden_passages = []
        for p in tmp:
            if p not in golden_passages:
                golden_passages.append(p)
        val["golden_passages"] = golden_passages
        val["type"] = data["type"]
        new_dataset.append(val)
        if data["type"] not in type_to_dataset:
            type_to_dataset[data["type"]] = []
        type_to_dataset[data["type"]].append(val)
    ret = {"total": new_dataset}
    ret.update(type_to_dataset)
    return ret


def load_default_format_data(data_path):
    filename = data_path.split("/")[-1]
    assert filename.endswith(".json"), f"Need json data: {data_path}"
    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    for did, data in enumerate(dataset):
        assert "question" in data, f"\"question\" not in data, {data_path}"
        question = data["question"]
        assert type(question) == str, f"\"question\": {question} should be a string"
        assert "answer" in data, f"\"answer\" not in data, {data_path}"
        answer = data["answer"]
        assert type(answer) == str or \
               (type(answer) == list and (not any(type(a) != str for a in answer))), \
               f"\"answer\": {answer} should be a string or a list[str]" 
        data["test_id"] = did
    return {filename: dataset}
    

def main(args):
    # output_dir = os.path.join(ROOT_DIR, "data_aug_dpr", args.dataset)
    # docs_file = os.path.join(ROOT_DIR, "all_docs_dpr.json")
    output_dir = os.path.join(ROOT_DIR, "FT_data", args.dataset)
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
            passage_ids, passages = bm25_retrieve(data["question"], topk=args.topk+10)
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
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--topk", type=int, default=3) 
    args = parser.parse_args()
    print(args)
    main(args)