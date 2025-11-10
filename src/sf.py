import os
import json
import argparse
from tqdm import tqdm
from utils import get_model, model_generate

ROOT_DIR = "/data-share/yeesuanAI08/zhanghanwen/D-PRAG"
INPUT_FILE = os.path.join(ROOT_DIR, "test.json")
OUTPUT_FILE = os.path.join(ROOT_DIR, "doc_aug", "test_slot.json")

# ======= Slot filling prompt =======
slot_filling_prompt = "I will provide a passage of text, and you need to extract two slot-filling examples from it. \
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
This list should have at least two elements, and each element should correspond to a different subject-relation pair.\
Passage:\n\
{passage}"

def get_sf(passage, model_name, model=None, tokenizer=None, generation_config=None):
    try_times = 20
    prompt = slot_filling_prompt.format(passage=passage)
    while try_times:
        output = model_generate(prompt, model, tokenizer, generation_config)
        try:
            sf = json.loads(output[output.find("["): output.rfind("]")+1])
            if len(sf) >= 2:
                for data in sf:
                    if "input" not in data or "output" not in data or "template_question" not in data:
                        return False, sf
                    if "[SEP]" not in data["input"]:
                        return False, sf
                return True, sf
        except Exception:
            try_times -= 1
    return False, output


def main(args):
    with open(INPUT_FILE, "r") as fin:
        docs = json.load(fin)

    model, tokenizer, _ = get_model(args.model_name)
    generation_config = dict(
        max_new_tokens=512,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        temperature=0.7,
        top_k=50,
    )

    ret = []

    for doc in tqdm(docs, total=len(docs)):
        passage = doc["text"]
        ok, aug_sf = get_sf(passage, args.model_name, model, tokenizer, generation_config)
        if not ok:
            continue
        val = {
            "global_id": doc["global_id"],
            "text": passage,
            "slot_filling": aug_sf
        }
        ret.append(val)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as fout:
        json.dump(ret, fout, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama3-8b-instruct", help="model name")
    args = parser.parse_args()
    print(args)
    main(args)
