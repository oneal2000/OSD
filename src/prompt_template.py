# this script provides prompt for encode and inference
# almost same as https://github.com/oneal2000/PRAG
import os
from root_dir_path import ROOT_DIR

current_dataset = None
fewshot = None
fewshot_path = os.path.join(ROOT_DIR, "src", "fewshot")

PROMPT_FC = "You are tasked with verifying a claim using the knowledge provided below combined with your own knowledge.\n\
Your response MUST be exactly one word: 'SUPPORTS' or 'REFUTES'. Do not output anything else and do not explain your choice.\n\
Example:\n\
Claim: The Eiffel Tower is located in Paris.\n\
Output: SUPPORTS\n\n\
Using the passages and your knowledge, output 'SUPPORTS' if the claim is true, or 'REFUTES' if the claim is false.\n\
{passages}\n\n\
Claim: {input}"


PROMPT_FC_LLM = "You are tasked with verifying a claim using only your own knowledge.\n\
Your response MUST be exactly one word: 'SUPPORTS' or 'REFUTES'. Do not output anything else and do not explain your choice.\n\
Example:\n\
Claim: The Eiffel Tower is located in Paris.\n\
Output: SUPPORTS\n\n\
Based on your knowledge, output 'SUPPORTS' if the claim is true, or 'REFUTES' if the claim is false.\n\
Claim: {input}"

PROMPT_SF = "You are tasked with extracting the object entity that completes a given slot using the knowledge provided below together with your own knowledge.\n\
The input format is: 'subject_entity [SEP] relation'.\n\
You must return only the object entity that is directly connected to the subject_entity through the specified relation.\n\
The extracted entity should be the one that can directly serve as the answer to the question:\n\
{template_question}\n\
Do not provide explanations or additional text, only output the object entity.\n\
{passages}\n\n\
Input: {input}"


PROMPT_SF_LLM = "You are tasked with finding the object entity that completes a given slot using only your own knowledge.\n\
The input format is: 'subject_entity [SEP] relation'.\n\
You must return only the object entity that is directly connected to the subject_entity through the specified relation.\n\
The extracted entity should be the one that can directly serve as the answer to the question:\n\
{template_question}\n\
Do not provide explanations or additional text, only output the object entity.\n\
Input: {input}"


USER_PROMPT = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n\
{passages}\n\n\
Question: {question}"

USER_PROMPT_NO_PSG = "You should answer the question based on your own knowledge.\n\
Question: {question}"

USER_PROMPT_WITH_COT = "You should reference the knowledge provided below and combine it with your own knowledge to answer the question. Please follow the format of the example I provided above.\n\
Here are some examples about how to answer the questions.\n\
{fewshot}\
Here are some reference.\n\
{passages}\n\n\
Let's think step by step. Answer the questions in the same format as above.\n\
Question: {question}"

USER_PROMPT_WITH_COT_NO_PSG = "You should answer the question based on your own knowledge. Please follow the format of the example I provided above.\n\
Here are some examples about how to answer the questions.\n\
{fewshot}\
Let's think step by step. Answer the questions in the same format as above.\n\
Question: {question}"

ASSISTANT_PROMPT = "The answer is {answer}"
ASSISTANT_PROMPT_WITH_COT = "Answer: {answer}"

ASSISTANT_PROMPT_OUTPUT = "Output: {output}"

def _get_prompt(question, passages=None, answer=None):
    question = question.strip()
    if not question.endswith('?'):
        question = question.strip() + '?'
    elif question.endswith(' ?'):
        question = (question[:-1]).strip() + '?'
     
    if passages and not isinstance(passages, list):
        passages = [passages]
    
    if answer is None:
        answer = ""
    else:
        answer = answer.strip()
        if not answer.endswith('.'):
            answer += "."
    return question, passages, answer


def get_fewshot(dataset):
    import json
    global current_dataset
    global fewshot
    # assert current_dataset is None
    if dataset.endswith("_golden"):
        dataset = dataset.split("_golden")[0]
    current_dataset = dataset
    with open(os.path.join(fewshot_path, dataset + ".json"), "r") as fin:
        tmp = json.load(fin)
    fewshot = ""
    for data in tmp:
        q = data["question"]
        a = data["answer"]
        fewshot += f"Question: {q}\nAnswer: {a}\n\n"


def get_prompt(tokenizer, question, passages=None, answer=None, with_cot=False):
    question, passages, answer = _get_prompt(question, passages, answer)
    # contexts = ""
    # if passages:
    #     for pid, psg in enumerate(passages):
    #         contexts += f"Passage {pid+1}: {psg}\n"
    contexts = ""
    if passages:
        for pid, psg in enumerate(passages):
            if isinstance(psg, dict):
                contexts += f"Passage {pid+1}: {psg['passage']}\n"
            else:
                contexts += f"Passage {pid+1}: {psg}\n"
    if not with_cot:
        user_content = USER_PROMPT.format(question=question, passages=contexts)
        assistant_content = ASSISTANT_PROMPT.format(answer=answer)
    else:
        assert fewshot is not None
        user_content = USER_PROMPT_WITH_COT.format(question=question, passages=contexts, fewshot=fewshot)
        assistant_content = ASSISTANT_PROMPT_WITH_COT.format(answer=answer)

    messages = [{
        "role": "user",
        "content": user_content,
    }]

    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True)
    inputs += tokenizer.encode(assistant_content, add_special_tokens=False)
    return inputs

def get_prompt_llm(tokenizer, question, answer=None, with_cot=False):
    question, _, answer = _get_prompt(question, None, answer)
    if not with_cot:
        user_content = USER_PROMPT_NO_PSG.format(question=question)
        assistant_content = ASSISTANT_PROMPT.format(answer=answer)
    else:
        assert fewshot is not None
        user_content = USER_PROMPT_WITH_COT_NO_PSG.format(question=question, fewshot=fewshot)
        assistant_content = ASSISTANT_PROMPT_WITH_COT.format(answer=answer)

    messages = [{
        "role": "user",
        "content": user_content,
    }]

    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True)
    inputs += tokenizer.encode(assistant_content, add_special_tokens=False)
    return inputs  

def get_prompt_fc(tokenizer, input, passages=None, output=None):
    input = input.strip()
    if not input.endswith('.'):
        input += "."
    elif input.endswith(' .'):
        input = (input[:-1]).strip() + '.'

    if output is None:
        output = ""

    if passages and not isinstance(passages, list):
        passages = [passages]

    contexts = ""
    if passages:
        for pid, psg in enumerate(passages):
            if isinstance(psg, dict):
                contexts += f"Passage {pid+1}: {psg['passage']}\n"
            else:
                contexts += f"Passage {pid+1}: {psg}\n"


    user_content = PROMPT_FC.format(passages=contexts, input = input)
    assistant_content = ASSISTANT_PROMPT_OUTPUT.format(output=output)

    messages = [{
        "role": "user",
        "content": user_content,
    }]
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True)
    inputs += tokenizer.encode(assistant_content, add_special_tokens=False)
    return inputs

def get_prompt_fc_llm(tokenizer, input, output=None):
    input = input.strip()
    if not input.endswith('.'):
        input += "."
    elif input.endswith(' .'):
        input = (input[:-1]).strip() + '.'

    if output is None:
        output = ""

    user_content = PROMPT_FC_LLM.format(input = input)
    assistant_content = ASSISTANT_PROMPT_OUTPUT.format(output=output)

    messages = [{
        "role": "user",
        "content": user_content,
    }]
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True)
    inputs += tokenizer.encode(assistant_content, add_special_tokens=False)
    return inputs

def get_prompt_sf(tokenizer, input, template_question, passages=None, output=None):
    # print(type(template_question), template_question)
    template_question = template_question.strip()
    if not template_question.endswith('?'):
        template_question += "?"
    elif template_question.endswith(' ?'):
        template_question = (template_question[:-1]).strip() + '?'

    if output is None:
        output = ""

    if passages and not isinstance(passages, list):
        passages = [passages]

    contexts = ""
    if passages:
        for pid, psg in enumerate(passages):
            if isinstance(psg, dict):
                contexts += f"Passage {pid+1}: {psg['passage']}\n"
            else:
                contexts += f"Passage {pid+1}: {psg}\n"

    user_content = PROMPT_SF.format(template_question=template_question, passages=contexts, input = input)
    assistant_content = ASSISTANT_PROMPT_OUTPUT.format(output=output)

    messages = [{
        "role": "user",
        "content": user_content,
    }]
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True)
    inputs += tokenizer.encode(assistant_content, add_special_tokens=False)
    return inputs

def get_prompt_sf_llm(tokenizer, input, template_question, output=None):
    template_question = template_question.strip()
    if not template_question.endswith('?'):
        template_question += "?"
    elif template_question.endswith(' ?'):
        template_question = (template_question[:-1]).strip() + '?'

    if output is None:
        output = ""

    user_content = PROMPT_SF_LLM.format(template_question=template_question, input = input)
    assistant_content = ASSISTANT_PROMPT_OUTPUT.format(output=output)

    messages = [{
        "role": "user",
        "content": user_content,
    }]
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True)
    inputs += tokenizer.encode(assistant_content, add_special_tokens=False)
    return inputs