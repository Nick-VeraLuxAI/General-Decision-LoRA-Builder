# === Run Command ===
# python decision_enricher.py legendary_decision_bank_enhanced.jsonl enriched_output.jsonl --add 25000


import json
import random

# 🎯 Target topic distribution
TARGET_TOPIC_DISTRIBUTION = {
    "productivity": 0.25,
    "emotions": 0.20,
    "habits": 0.15,
    "interactions": 0.15,
    "learning": 0.10,
    "tools": 0.10,
    "coffee": 0.05
}

# Enriched topic base phrases, will auto-expand to create diversity
ENRICHED_BANK = {
    "productivity": [
        "I like working under pressure, but only when I have some control over the outcome.",
        "You take on too much, then excel through chaos.",
        "I'm most productive when there's a tight deadline.",
        "Multitasking is energizing, but sometimes leads to burnout."
    ],
    "emotions": [
        "I usually avoid conflict, even when I know I should speak up.",
        "Sometimes, I suppress my feelings to keep the peace.",
        "Expressing gratitude helps me recover from setbacks.",
        "Stress makes me overthink simple things."
    ],
    "habits": [
        "I once worked 100-hour weeks, and I’m still trying to unlearn that habit.",
        "Waking up early is tough, but it sets a positive tone for my day.",
        "I tend to snack when I’m bored.",
        "Daily journaling is a habit I wish I had started sooner."
    ],
    "interactions": [
        "I prefer direct feedback, though it sometimes makes me defensive at first.",
        "I often listen more than I speak in group settings.",
        "You find it easier to communicate by email than in person.",
        "Collaborating with others boosts my creativity."
    ],
    "learning": [
        "You learn fast, but sometimes burn out trying to do it all at once.",
        "I retain information better when I teach it to others.",
        "Mistakes help me learn more than successes.",
        "Curiosity drives me to explore new fields."
    ],
    "tools": [
        "I enjoy learning new keyboard shortcuts.",
        "Using productivity apps helps me stay on track.",
        "Automation tools save me hours each week.",
        "Spreadsheets are my secret weapon for organization."
    ],
    "coffee": [
        "Strong coffee is how I start every day.",
        "I can’t focus until I’ve had my morning espresso.",
        "I drink coffee for the ritual, not just the caffeine.",
        "Cutting back on coffee is always a challenge."
    ]
}

DECISION_OPTIONS = ["Store this in long-term memory", "Discard it for now"]

REASONING_TEMPLATES = [
    "This reflects a topic of interest ({topic}) and could influence future choices or behavior.",
    "Recording this {topic} observation may improve future self-awareness.",
    "This is a good example of how {topic} shapes my routines or reactions.",
    "Keeping this in mind could help me adjust my {topic} strategies.",
    "This seems less relevant to future decisions, but could still provide context.",
    "Given its sentiment ({sentiment}), it may or may not be important for long-term memory."
]

def classify_sentiment(text):
    # Improved, now picks up more nuance
    text_lower = text.lower()
    if any(w in text_lower for w in ["but", "though", "even", "although", "despite", "unless"]):
        return "mixed"
    elif any(w in text_lower for w in ["love", "enjoy", "like", "prefer", "thrive", "energizing", "positive", "boosts"]):
        return "positive"
    elif any(w in text_lower for w in ["fear", "avoid", "fail", "burn", "burnout", "suppress", "tough", "stress", "snack", "challenge"]):
        return "negative"
    else:
        return "neutral"

def randomize_choice(sentiment):
    # More likely to store positive/neutral, discard negative/mixed, but randomize
    if sentiment == "positive":
        return random.choices(DECISION_OPTIONS, [0.9, 0.1])[0]
    elif sentiment == "negative":
        return random.choices(DECISION_OPTIONS, [0.3, 0.7])[0]
    elif sentiment == "mixed":
        return random.choices(DECISION_OPTIONS, [0.5, 0.5])[0]
    else:
        return random.choices(DECISION_OPTIONS, [0.6, 0.4])[0]

def random_confidence(choice):
    # Store = higher, Discard = lower, but randomized
    if choice == "Store this in long-term memory":
        return round(random.uniform(0.8, 0.99), 2)
    else:
        return round(random.uniform(0.4, 0.8), 2)

def random_importance(choice, sentiment):
    # Store+positive = higher, discard+negative = lower, mixed varies
    if choice == "Store this in long-term memory" and sentiment == "positive":
        return random.randint(4, 5)
    elif choice == "Discard it for now" and sentiment == "negative":
        return random.randint(1, 3)
    elif sentiment == "mixed":
        return random.randint(2, 4)
    else:
        return random.randint(2, 5)

def random_reasoning(topic, sentiment):
    template = random.choice(REASONING_TEMPLATES)
    return template.format(topic=topic, sentiment=sentiment)

def augment_text(text):
    # Optionally, further randomize or paraphrase base phrases
    swaps = [
        ("I ", "You "),
        ("You ", "I "),
        ("is", "can be"),
        ("makes me", "leads me to"),
        ("sometimes", "occasionally"),
        ("but", "yet"),
        ("every day", "each morning"),
        ("start", "kick off"),
    ]
    # 50% chance to do a swap
    for old, new in swaps:
        if random.random() < 0.2:
            text = text.replace(old, new)
    return text

def generate_enriched_decisions(n=25000):
    enriched = []
    per_topic = {t: int(n * ratio) for t, ratio in TARGET_TOPIC_DISTRIBUTION.items()}
    for topic, count in per_topic.items():
        base_phrases = ENRICHED_BANK[topic]
        for _ in range(count):
            text = augment_text(random.choice(base_phrases))
            sentiment = classify_sentiment(text)
            choice = randomize_choice(sentiment)
            confidence = random_confidence(choice)
            importance_score = random_importance(choice, sentiment)
            reasoning = random_reasoning(topic, sentiment)
            enriched.append({
                "situation": text,
                "options": DECISION_OPTIONS,
                "choice": choice,
                "reasoning": reasoning,
                "confidence": confidence,
                "importance_score": importance_score,
                "topic": topic,
                "sentiment": sentiment,
                "type": "observation"
            })
    return enriched

# === File helpers unchanged ===
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# === Main execution ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to base balanced JSONL file")
    parser.add_argument("output", help="Path to save enriched JSONL file")
    parser.add_argument("--add", type=int, default=25000, help="Number of enriched entries to add")
    args = parser.parse_args()

    base = load_jsonl(args.input)
    enriched = generate_enriched_decisions(args.add)
    final = base + enriched
    random.shuffle(final)
    save_jsonl(final, args.output)

    print(f"✨ Added {len(enriched)} enriched decision entries across {len(TARGET_TOPIC_DISTRIBUTION)} topics.")
    print(f"🎯 Final dataset size: {len(final)} → saved to {args.output}")

