# === Run Command ===
# python decision_cleaner.py enriched_output.jsonl



import json
import sys

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def clean_decisions(data, dedupe_on="situation"):
    seen = set()
    output = []
    dropped = 0

    for d in data:
        situation = d.get("situation", "").strip().casefold()
        if not situation:
            dropped += 1
            continue

        # Set deduplication signature
        if dedupe_on == "all":
            sig = (
                situation,
                d.get("choice", "").strip().casefold(),
                d.get("reasoning", "").strip().casefold(),
            )
        else:
            sig = situation

        if sig in seen:
            dropped += 1
            continue
        seen.add(sig)
        output.append(d)
    print(f"✅ De-duplicated: {len(data)} → {len(output)} kept, {dropped} dropped.")
    return output

if __name__ == "__main__":
    # USAGE:
    #   python decision_cleaner.py input.jsonl [output.jsonl] [dedupe_on]
    # EXAMPLES:
    #   python decision_cleaner.py enriched_output.jsonl
    #   python decision_cleaner.py enriched_output.jsonl cleaned_output.jsonl
    #   python decision_cleaner.py enriched_output.jsonl cleaned_output.jsonl all
    input_path = sys.argv[1] if len(sys.argv) > 1 else "enriched_output.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "cleaned_output.jsonl"
    dedupe_on = sys.argv[3] if len(sys.argv) > 3 else "situation"  # or "all"

    raw = load_jsonl(input_path)
    cleaned = clean_decisions(raw, dedupe_on=dedupe_on)
    save_jsonl(cleaned, output_path)
    print(f"✅ Cleaned {len(cleaned)} unique entries → saved to {output_path}")
