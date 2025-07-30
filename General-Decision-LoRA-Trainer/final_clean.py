# === Run Command ===
# python final_clean.py enriched_100k_output.jsonl legendary_final_output.jsonl


import json
import random
import sys

# You can expand these lists if you add more types/sentiments in your data.
TYPES = ["preference", "fact", "observation"]
SENTIMENTS = ["positive", "negative", "mixed", "uncertain"]

def fill_fields(entry):
    if not entry.get("type"):
        entry["type"] = random.choice(TYPES)
    if not entry.get("sentiment"):
        entry["sentiment"] = random.choice(SENTIMENTS)
    return entry

def main(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        count = 0
        for line in infile:
            if not line.strip():
                continue
            entry = json.loads(line)
            entry = fill_fields(entry)
            outfile.write(json.dumps(entry) + '\n')
            count += 1
    print(f"✅ Cleaned {count} entries and filled all missing fields!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python final_clean.py input.jsonl output.jsonl")
    else:
        main(sys.argv[1], sys.argv[2])
