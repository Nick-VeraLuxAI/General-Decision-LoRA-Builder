# === Start Server ====
#CUDA_VISIBLE_DEVICES=0 ~/Desktop/Decision-Lora-Builder/llama.cpp/build/bin/llama-server \
#  -m /home/ndesantis/Desktop/General-Decision-Lora-Builder/Meta-Llama-3-8B-Instruct.Q6_K.gguf \
#  --port 8080 \
#  --ctx-size 4096 \
#  --n-gpu-layers 256


#CUDA_VISIBLE_DEVICES=0 ~/Desktop/Decision-Lora-Builder/llama.cpp/build/bin/llama-server \
#  -m /home/ndesantis/Desktop/General-Decision-Lora-Builder/Meta-Llama-3-8B-Instruct.Q6_K.gguf \
#  --port 8082 \
#  --ctx-size 4096 \
#  --n-gpu-layers 256

#CUDA_VISIBLE_DEVICES=1 ~/Desktop/Decision-Lora-Builder/llama.cpp/build/bin/llama-server \
#  -m /home/ndesantis/Desktop/General-Decision-Lora-Builder/Meta-Llama-3-8B-Instruct.Q6_K.gguf \
#  --port 8081 \
#  --ctx-size 4096 \
#  --n-gpu-layers 256

#CUDA_VISIBLE_DEVICES=1 ~/Desktop/Decision-Lora-Builder/llama.cpp/build/bin/llama-server \
#  -m /home/ndesantis/Desktop/General-Decision-Lora-Builder/Meta-Llama-3-8B-Instruct.Q6_K.gguf \
#  --port 8083 \
#  --ctx-size 4096 \
#  --n-gpu-layers 256


# === Run Script Command ===
#cd /home/ndesantis/Desktop/General-Decision-Lora-Builder/JSONL-Generation-Engine
#python3 decision_bank_generator.py --count 334 --threads 8 --output legendary_decision_bank.jsonl



import json
import time
import random
import argparse
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading

API_URLS = [
    "http://localhost:8080/v1/chat/completions",
    "http://localhost:8081/v1/chat/completions",
    "http://localhost:8082/v1/chat/completions",
    "http://localhost:8083/v1/chat/completions"
    # ...add more if you launch more servers
]

HEADERS = {"Content-Type": "application/json"}
MAX_RETRIES = 3
DEFAULT_OUTPUT_DIR = "./content"

SEED_TOPICS = [
    "Workplace conflict", "Parenting dilemma", "Travel choice", "Health decision",
    "Financial risk", "Moral dilemma", "Unexpected opportunity", "Disaster scenario"
]

# === Set this to your model for the tokenizer analysis ===
model_name = "codellama/CodeLlama-13b-hf"

def make_prompt(seed_topic=None):
    topic_line = f"Topic: {seed_topic}\n" if seed_topic else ""
    return f"""{topic_line}You are an AI assistant tasked with inventing a realistic, specific decision scenario and answering it.

Generate a JSON object describing:
- A situation requiring a real-life decision.
- Two or three reasonable options.
- Your chosen option.
- A short, clear, human-like explanation of the reasoning.
- A confidence score (0.70–0.99).

IMPORTANT:
*ONLY* output the JSON object. Do NOT add any explanation, commentary, markdown, or preamble. Do NOT write anything outside the JSON.

Valid JSON format:
{{
  "situation": "<describe a realistic, specific decision scenario>",
  "options": ["<option 1>", "<option 2>", "(optional) <option 3>"],
  "choice": "<copy one option exactly>",
  "reasoning": "<human-style, well-phrased explanation>",
  "confidence": <float between 0.70 and 0.99>
}}
"""

def extract_json(content):
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        snippet = content[start:end+1]
        try:
            parsed = json.loads(snippet)
            return parsed
        except Exception as e:
            print(f"⚠️ JSON extraction failed: {e}")
    return None

def get_decision(seed_topic=None, api_url=None):
    prompt = make_prompt(seed_topic)
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 400
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(api_url, json=payload, headers=HEADERS, timeout=60)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                parsed = None
                try:
                    parsed = json.loads(content)
                except Exception:
                    parsed = extract_json(content)
                if parsed and all(k in parsed for k in ["situation", "options", "choice", "reasoning", "confidence"]):
                    return parsed
                else:
                    print(f"⚠️ Invalid JSON (attempt {attempt}): {content[:120]}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:120]}")
        except Exception as e:
            print(f"⚠️ Exception on attempt {attempt}: {e}")
        time.sleep(1 + random.uniform(0, 1))
    return None

def worker(args):
    seed_topic, api_url = args
    return get_decision(seed_topic, api_url)

def analyze_lengths(jsonl_path):
    from transformers import AutoTokenizer
    import matplotlib.pyplot as plt

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    lengths = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            prompt = (
                f"Situation:\n\"{entry['situation']}\"\n\n"
                f"Options:\n- " + "\n- ".join(entry['options']) +
                f"\n\nWhat should be chosen and why?\n\n### Response:"
            )
            response = json.dumps({
                "choice": entry["choice"],
                "reasoning": entry["reasoning"],
                "confidence": entry["confidence"]
            })
            combined = prompt + " " + response
            lengths.append(len(tokenizer(combined)["input_ids"]))

    plt.hist(lengths, bins=50)
    plt.title("Prompt+Response Token Lengths")
    plt.xlabel("Number of tokens")
    plt.ylabel("Count")
    plt.show()
    print("Max length:", max(lengths))
    print("Average length:", sum(lengths) // len(lengths))
    print("95th percentile:", sorted(lengths)[int(0.95 * len(lengths))])
    print("99th percentile:", sorted(lengths)[int(0.99 * len(lengths))])

def main():
    parser = argparse.ArgumentParser(description="Generate general decision JSONL using LLM")
    parser.add_argument("--count", type=int, default=10000, help="Number of decisions to generate")
    parser.add_argument("--threads", type=int, default=24, help="Parallel threads")
    parser.add_argument("--output", type=str, help="Custom output filename (.jsonl)")
    parser.add_argument("--analyze", action="store_true", help="Analyze max token lengths after generation")
    args = parser.parse_args()

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f"general_decisions_{timestamp}.jsonl")

    print(f"🔁 Generating {args.count} general decisions using {args.threads} threads...")
    print(f"💾 Output: {output_file}\n")

    with open(output_file, "w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = []
            for i in range(args.count):
                seed_topic = random.choice(SEED_TOPICS) if SEED_TOPICS and random.random() < 0.8 else None
                api_url = API_URLS[i % len(API_URLS)]
                futures.append(executor.submit(worker, (seed_topic, api_url)))

            i = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    i += 1
                    if i % 100 == 0:
                        print(f"[{i}/{args.count}] ✅")

    print(f"\n✅ All done. Output saved to: {output_file}")

    # === Max length analysis ===
    if args.analyze:
        print("\n🔎 Running max-length/token analysis...")
        analyze_lengths(output_file)

if __name__ == "__main__":
    main()
