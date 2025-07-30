# === Run Command ===
# python3 decision_reasoning_enhancer.py /home/ndesantis/Desktop/General-Decision-Lora-Builder/JSONL-Generation-Engine/legendary_decision_bank.jsonl /home/ndesantis/Desktop/General-Decision-Lora-Builder/JSONL-Refinement-Engine/legendary_decision_bank_enhanced.jsonl --workers 24

import json
import random
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def enhance_reasoning(entry):
    choice = entry.get("choice", "").strip()
    situation = entry.get("situation", "").strip().lower()

    # (Same enhancement logic as before...)
    if "productivity" in situation or "work" in situation:
        reasoning = "This memory may affect future productivity strategies, work habits, or self-management."
        category = "productivity"
        importance = 9 if choice.startswith("Store") else 5
    elif "feeling" in situation or "emotion" in situation or "scared" in situation or "excited" in situation:
        reasoning = "This emotional context may influence future motivation, mood tracking, or behavioral patterns."
        category = "emotions"
        importance = 8 if choice.startswith("Store") else 4
    elif "habit" in situation or "routine" in situation:
        reasoning = "This highlights a pattern or routine, which could inform long-term behavior modeling."
        category = "habits"
        importance = 9 if choice.startswith("Store") else 5
    elif "person" in situation or "client" in situation or "team" in situation:
        reasoning = "This involves interpersonal dynamics that may impact communication or collaboration."
        category = "interactions"
        importance = 8 if choice.startswith("Store") else 4
    elif "learn" in situation or "realization" in situation:
        reasoning = "This reflects a learning or insight moment that may be valuable for future decisions."
        category = "learning"
        importance = 9 if choice.startswith("Store") else 6
    elif "coffee" in situation:
        reasoning = "This expresses a personal preference, which may or may not affect behavior patterns."
        category = "coffee"
        importance = 6 if choice.startswith("Store") else 3
    else:
        reasoning = entry.get("reasoning", "No reasoning provided.").strip()
        category = "misc"
        importance = 7 if choice.startswith("Store") else 4

    return {
        **entry,
        "reasoning": reasoning,
        "importance_score": importance,
        "memory_category": category
    }

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to raw decision dataset (.jsonl)")
    parser.add_argument("output", help="Path to save enhanced dataset (.jsonl)")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of worker processes")
    args = parser.parse_args()

    print(f"📥 Loading: {args.input}")
    data = load_jsonl(args.input)

    print(f"🔁 Enhancing {len(data)} entries with {args.workers} processes...")
    with Pool(args.workers) as pool:
        enhanced = list(tqdm(pool.imap(enhance_reasoning, data, chunksize=1000), total=len(data)))

    save_jsonl(enhanced, args.output)
    print(f"✅ Saved enhanced dataset to: {args.output}")

if __name__ == "__main__":
    main()
