import json
from tqdm import tqdm

input_file = "kilt.jsonl" 
output_file = "kilt_pre.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in tqdm(fin, total=total_lines, desc="Processing"):
        parts = line.strip().split("\t", 1)
        if len(parts) != 2:
            continue
        global_id, json_str = parts
        data = json.loads(json_str)

        new_item = {
            "global_id": int(global_id),
            "wikipedia_id": data.get("wikipedia_id"),
            "wikipedia_title": data.get("wikipedia_title"),
            "text": data.get("text"),
        }

        fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
