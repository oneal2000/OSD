import os
import json
import re
import random
import argparse
from tqdm import tqdm
from utils import get_model, model_generate


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


pubmedqa_prompt_template = (
    "I will provide a passage. "
    "Based ONLY on the factual content of the passage, generate four yes/no questions.\n\n"
    "Each question must be objectively answerable as 'yes' or 'no' based on the passage.\n"
    "For each question, provide the correct answer as exactly one lowercase word: 'yes' or 'no'.\n\n"
    "The output format must be a JSON list like this:\n"
    "[\n"
    "  {{\n"
    "    \"question\": \"...\",\n"
    "    \"answer\": \"yes\"\n"
    "  }}\n"
    "]\n\n"
    "This list must contain at least four elements.\n\n"
    "Passage:\n{passage}\n"
)


dialogue_prompt = "I will provide a passage of text from Wikipedia, and you need to generate four knowledge-grounded dialogues in the style of Wizard of Wikipedia dataset. \
Each dialogue should be a natural, multi-turn conversation between a curious user and a knowledgeable assistant (wizard) who has access to the provided passage. \
The assistant should provide informative, detailed responses based on the passage content, while maintaining a natural conversational flow. \
The input should contain the conversation history (alternating user and assistant messages), and the output should be the assistant's response to the last user message. \
Each message in the input should be separated by a newline character (\\n), except the last message. \
The assistant's responses should be informative, engaging, and naturally incorporate information from the passage without directly copying it. \
The user's questions should progressively explore different aspects of the topic, building upon previous turns in the conversation. \
\nYou need to generate the dialogues in the following format: \n\
[\n\
    {{\n\
        \"input\": \"hello, i like spicy food .\\nI do too especially chili peppers. They're widely used in many different types of food to add spiciness.\\nis chili pepper a specific pepper or is it a general term for all peppers\\nIt's a specific pepper originated in Mexico\\nthat make sense\\nYou'd think Dr Pepper was spicy since it has pepper in the name but is has a different unique flavor.\\nyou know I never thought about that, there should be a spicy soda\",\n\
        \"output\": \"Jones soda might have a spicy soda, they're known for their unusual flavors.\"\n\
    }},\n\
]\n\n\
Important guidelines:\n\
- The assistant should provide detailed, informative responses based on the passage\n\
- The conversation should feel natural and engaging, not robotic\n\
- Each dialogue should have varied number of turns (typically 1-6 turns)\n\
- The user's questions should explore different aspects of the topic mentioned in the passage\n\
- The assistant's responses should synthesize information from the passage in a natural way\n\
- Generate at least four dialogues with different conversation flows\n\
\nYou only need to output this list in the above format.\n\
This list should have at least four elements.\n\
Passage:\n\
{passage}"



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


def get_pubmedqa(passage, model_name, model=None, tokenizer=None, generation_config=None):
    try_times = 100
    prompt = pubmedqa_prompt_template.format(passage=passage)
    output = None

    while try_times:
        output = model_generate(prompt, model, tokenizer, generation_config)
        output = fix_json(output, model_name)

        try:
            data = json.loads(output)
            if len(data) >= 4:
                data = data[:4]
                for item in data:
                    if "question" not in item or "answer" not in item:
                        return False, data
                    if item["answer"] not in ["yes", "no"]:
                        return False, data
                return True, data
        except:
            try_times -= 1
            print(output)
            print(passage)

    return False, output


def get_dialogue(passage, model=None, tokenizer=None, generation_config=None):
    try_times = 100
    prompt = dialogue_prompt.format(passage=passage)
    output = None
    while try_times:
        output = model_generate(prompt, model, tokenizer, generation_config)
        output = re.sub(r",\s*([\]}])", r"\1", output)
        try:
            dialogues = json.loads(output[output.find("["): output.rfind("]")+1])
            if len(dialogues) >= 4:
                dialogues = dialogues[:4]
                for data in dialogues:
                    if "input" not in data or "output" not in data:
                        return False, dialogues
                return True, dialogues
        except:
            print(output)
            print(passage)
            try_times -= 1
    return False, output
    

def main(args):
    with open(args.input_file, "r") as fin:
        docs = json.load(fin)

    input_basename = os.path.basename(args.input_file)
    is_med_file = input_basename == "all_docs_med.json"

    processed = {}
    if os.path.exists(args.output_file):
        if os.path.getsize(args.output_file) == 0:
            processed_list = []
        else:
            with open(args.output_file, "r") as fin:
                processed_list = json.load(fin)
        processed = {d["global_id"]: d for d in processed_list}
        print(f" {len(processed)} Processed")
    
    model, tokenizer, _ = get_model(args.model_name)
    generation_config = dict(
        max_new_tokens=1024,
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
        passage = doc["text"]

        if is_med_file:
            doc_rewrite = get_rewrite(passage, args.model_name, model, tokenizer, generation_config)
            aug_pubmedqa = get_pubmedqa(passage, args.model_name, model, tokenizer, generation_config)
            if aug_pubmedqa[0] == False:
                continue
            aug_result = {"rewrite": doc_rewrite, "pubmedqa": aug_pubmedqa[1][:3]}
            doc["augment"] = [aug_result]
            doc["task"] = [{
                "type": "pubmedqa",
                "data": aug_pubmedqa[1][3]
            }]
        else:
            doc_rewrite = get_rewrite(passage, args.model_name, model, tokenizer, generation_config)
            aug_qa = get_qa(passage, args.model_name, model, tokenizer, generation_config)
            if fix_qa(aug_qa)[0] == False:
                continue
            aug_fc = get_fc(passage, args.model_name, model, tokenizer, generation_config)
            if aug_fc[0] == False:
                continue
            aug_sf = get_sf(passage, model, tokenizer, generation_config)
            if aug_sf[0] == False:
                continue
            aug_dialogue = get_dialogue(passage, model, tokenizer, generation_config)
            if aug_dialogue[0] == False:
                continue
            aug_result = {
                "rewrite": doc_rewrite,
                "qa": aug_qa[:3],
                "fact_checking": aug_fc[1][:3],
                "slot_filling": aug_sf[1][:3],
                "dialogue": aug_dialogue[1][:3],
            }
            doc["augment"] = [aug_result]
            doc["task"] = [
                {
                    "type": "fact_checking",
                    "data": aug_fc[1][3]
                },
                {
                    "type": "slot_filling",
                    "data": aug_sf[1][3]
                },
                {
                    "type": "open_domain_qa",
                    "data": [{
                        "input": aug_qa[3]["question"],
                        "output": aug_qa[3]["answer"],
                        "full_answer": aug_qa[3]["full_answer"]
                    }]
                },
                {
                    "type": "dialogue",
                    "data": aug_dialogue[1][3]
                },
            ]
        val = {
            "global_id": doc["global_id"],
            "passage": passage,
            "augment": doc["augment"],
            "task": doc["task"]
        }
        ret.append(val)
        pbar.update(1)

        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

        with open(args.output_file, "w") as fout:
            json.dump(ret, fout, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama3-8b-instruct", help="model name")
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()
    print(args)
    main(args)