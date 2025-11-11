# this script is to augment documents with rewriting, QA, fact-checking, and slot-filling using LLMs
# the format of the augmented data is as follows: rewrite + 3 qa + 3 fact-checking + 3 slot-filling + task data (1 fact-checking + 1 slot-filling + 1 open-domain QA)
# almost same as https://github.com/oneal2000/PRAG
import os
import json
import re
import random
import argparse
from tqdm import tqdm
from utils import get_model, model_generate
from root_dir_path import ROOT_DIR

INPUT_FILE = os.path.join(ROOT_DIR, "all_docs_dpr.json")
OUTPUT_FILE = os.path.join(ROOT_DIR, "doc_aug", "dpr_3.json")


random.seed(42)

qa_prompt_template = "I will provide a passage of text, and you need to generate four different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.\n\
You need to generate the question and answer in the following format:\n\
[\n\
    {{\n\
        \"question\": \"What is the capital of France?\",\n\
        \"answer\": \"Paris\"\n\
        \"full_answer\": \"The capital of France is Paris.\"\n\
    }}, \n\
]\n\n\
This list should have at least four elements. You only need to output this list in the above format.\n\
Passage:\n\
{passage}"


fact_checking_prompt = "I will provide a passage of text, and you need to generate four claims based on the content of this passage. Each claim should be verifiable using the information provided in the passage. Additionally, please provide an appropriate label for each claim, indicating whether it is 'SUPPORTS' or 'REFUTES'\n\
You need to generate each claim and label in the following format:\n\
[\n\
    {{\n\
        \"input\": \"The capital of France is Paris.\",\n\
        \"output\": \"SUPPORTS\"\n\
    }}, \n\
]\n\n\
This list should have at least four elements. You only need to output this list in the above format.\n\
Passage:\n\
{passage}"


slot_filling_prompt = "I will provide a passage of text, and you need to extract four slot-filling examples from it. \
Each example should identify a subject entity mentioned in the passage, one of its relations, and the corresponding object entity. \
You should model the input as a structured string in the format 'subject_entity [SEP] relation'. \
The output should be the object entity that fills the slot, based on the passage. \
Additionally, for each slot-filling example, you need to generate a natural language template question that could be answered by the output (use the subject entity and relation in the question)  \
\nYou need to generate the input, output and the template question in the following format:\n\
[\n\
    {{\n\
        \"input\": \"Albert Einstein [SEP] educated_at\",\n\
        \"output\": \"ETH Zurich, University of Zurich\",\n\
        \"template_question\": \"Where did Albert Einstein receive education?\"\n\
    }},\n\
]\n\n\
You only need to output this list in the above format.\n\
This list should have at least four elements\
Passage:\n\
{passage}"


# od_qa_prompt = "I will provide a passage of text, and you need to generate two different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.\n\
# You need to generate the question and answer in the following format:\n\
# [\n\
#     {{\n\
#         \"input\": \"What is the capital of France?\",\n\
#         \"output\": \"Paris\"\n\
#     }}, \n\
# ]\n\n\
# This list should have at least two elements. You only need to output this list in the above format.\n\
# But do not generate questions that are similar to the given queations.\n\
# Passage:\n\
# {passage}\n\
# Questions:\n\
# {question}"


def fix_json(output, model_name):
        if model_name == "llama3.2-1b-instruct":
            if "[" in output:
                output = output[output.find("["):]
            if "]" in output:
                output = output[:output.rfind("]")+1]
            if output.endswith(","):
                output = output[:-1]
            if not output.endswith("]"):
                output += "]"
        elif model_name == "llama3-8b-instruct":
            if "[" in output:
                output = output[output.find("["):] 
            if "]" in output:
                output = output[:output.rfind("]")+1]
        return output


def get_rewrite(passage, model_name, model=None, tokenizer=None, generation_config=None):
    rewrite_prompt = "Rewrite the following passage. While keeping the entities, proper nouns, and key details such as names, locations, and terminology intact, create a new version of the text that expresses the same ideas in a different way. Make sure the revised passage is distinct from the original one, but preserves the core meaning and relevant information.\n{passage}"
    return model_generate(rewrite_prompt.format(passage=passage), model, tokenizer, generation_config)


def fix_qa(qa):
    if isinstance(qa, list):
        if len(qa) >= 4:
            qa = qa[:4]
            for data in qa:
                if "question" not in data or "answer" not in data or "full_answer" not in data:
                    return False, qa
                if isinstance(data["answer"], list):
                    data["answer"] = ", ".join(data["answer"])
                if isinstance(data["answer"], int):
                    data["answer"] = str(data["answer"])
                if data["answer"] is None:
                    data["answer"] = "Unknown"
            return True, qa
    return False, qa

def get_qa(passage, model_name, model=None, tokenizer=None, generation_config=None):
    try_times = 100
    prompt = qa_prompt_template.format(passage=passage)
    output = None
    while try_times:
        output = model_generate(prompt, model, tokenizer, generation_config)
        # print(output)
        output = fix_json(output, model_name)
        try:
            qa = json.loads(output)
            ret, qa = fix_qa(qa)
            if ret:
                return qa
        except:
            print(output)
            print(passage)
            try_times -= 1
    return output

def get_fc(passage, model_name, model=None, tokenizer=None, generation_config=None):
    try_times = 100
    prompt = fact_checking_prompt.format(passage=passage)
    output = None
    while try_times:
        output = model_generate(prompt, model, tokenizer, generation_config)
        output = fix_json(output, model_name)
        try:
            fc = json.loads(output)
            if len(fc) >= 4:
                fc = fc[:4]
                for data in fc:
                    if "input" not in data or "output" not in data:
                        return False, fc
                    if data["output"] not in ["SUPPORTS", "REFUTES"]:
                        return False, fc
                return True, fc
        except:
            try_times -= 1
            print(output)
            print(passage)
    return False, output

def get_sf(passage, model=None, tokenizer=None, generation_config=None):
    try_times = 100
    prompt = slot_filling_prompt.format(passage=passage)
    output = None
    while try_times:
        output = model_generate(prompt, model, tokenizer, generation_config)
        output = re.sub(r",\s*([\]}])", r"\1", output)
        try:
            sf = json.loads(output[output.find("["): output.rfind("]")+1])
            if len(sf) >= 4:
                sf = sf[:4]
                for data in sf:
                    if "input" not in data or "output" not in data or "template_question" not in data:
                        return False, sf
                    if "[SEP]" not in data["input"]:
                        return False, sf
                    if isinstance(data["output"], list):
                        data["output"] = ", ".join(data["output"])
                    if isinstance(data["output"], int):
                        data["output"] = str(data["output"])
                return True, sf
        except:
            print(output)
            print(passage)
            try_times -= 1
    return False, output

# def get_odqa(passage, question, model_name, model=None, tokenizer=None, generation_config=None):
#     try_times = 100
#     prompt = od_qa_prompt.format(passage=passage, question=question)
#     output = None
#     while try_times:
#         output = model_generate(prompt, model, tokenizer, generation_config)
#         output = fix_json(output, model_name)
#         try:
#             odqa = json.loads(output)
#             if len(odqa) >= 2:
#                 for data in odqa:
#                     if "input" not in data or "output" not in data:
#                         return False, odqa
#                     if isinstance(data["output"], list):
#                         data["output"] = ", ".join(data["output"])
#                     if isinstance(data["output"], int):
#                         data["output"] = str(data["output"])
#                     if data["output"] is None:
#                         data["output"] = "Unknown"
#                 return True, odqa
#         except:
#             try_times -= 1
#     return False, output
    

def main(args):
    with open(INPUT_FILE, "r") as fin:
        docs = json.load(fin)

    processed = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as fin:
            processed_list = json.load(fin)
            processed = {d["global_id"]: d for d in processed_list}
        print(f" {len(processed)} Processed")
    
    model, tokenizer, _ = get_model(args.model_name)
    generation_config = dict(
        max_new_tokens=512,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        temperature=0.7,
        top_k=50,
    )

    ret = list(processed.values())
    done_ids = set(processed.keys())

    pbar = tqdm(total=len(docs), initial=len(done_ids))

    for doc in docs:
        if doc["global_id"] in done_ids:
            continue
        # print(f"Processing doc {doc['global_id']}")
        doc["augment"] = []
        passage = doc["text"]
        doc_rewrite = get_rewrite(passage, args.model_name, model, tokenizer, generation_config)
        aug_qa = get_qa(passage, args.model_name, model, tokenizer, generation_config)
        if fix_qa(aug_qa)[0] == False: # skip error passage
            continue
        aug_fc = get_fc(passage, args.model_name, model, tokenizer, generation_config)
        if aug_fc[0] == False: # skip error passage
            continue
        aug_sf = get_sf(passage, model, tokenizer, generation_config)
        if aug_sf[0] == False: # skip error passage
            continue
        qa_for_aug = aug_qa[:3]
        fc_for_aug = aug_fc[1][:3]
        sf_for_aug = aug_sf[1][:3]
        doc["augment"].append({
                "rewrite": doc_rewrite,
                "qa": qa_for_aug,
                "fact_checking": fc_for_aug,
                "slot_filling": sf_for_aug
            })
        doc["task"] = []
        doc["task"].append({
            "type": "fact_checking",
            "data": aug_fc[1][3]
        })
        doc["task"].append({
            "type": "slot_filling",
            "data": aug_sf[1][3]
        })
        qa_for_task = aug_qa[3]
        open_domain_qa = [{
            "input": qa_for_task["question"],
            "output": qa_for_task["answer"],
            "full_answer": qa_for_task["full_answer"]
        }]
        doc["task"].append({
            "type": "open_domain_qa",
            "data": open_domain_qa
        })
        val = {
            "global_id": doc["global_id"],
            "passage": passage,
            "augment": doc["augment"],
            "task": doc["task"]
        }
        ret.append(val)
        pbar.update(1)

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        with open(OUTPUT_FILE, "w") as fout:
            json.dump(ret, fout, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama3-8b-instruct", help="model name")
    args = parser.parse_args()
    print(args)
    main(args)