# === Run Command ===
# python decision_balancer.py cleaned_output.jsonl enriched_100k_output.jsonl

import json
import random

DECISION_OPTIONS = [
    "Store this in long-term memory",
    "Discard it for now",
    "Flag for further review"
]

SCENARIO_SPECIFIC_OPTIONS = [
    ["Store for leadership training", "Ignore for now", "Ask for peer feedback"],
    ["Save for next review cycle", "Remove from record", "Escalate to supervisor"],
    ["Log as a potential risk", "Skip as irrelevant", "Mark as requires context"],
    ["Initiate a follow-up", "Send for audit", "Mark as unverified"],
    ["Forward to department head", "Archive for future review", "Tag as low priority"]
]

RARE_CASES = [
    "Unexpected server outage disrupted the daily routine.",
    "A team member submitted work under another's name.",
    "Disagreement about project scope led to escalation.",
    "The log file is corrupted and unreadable.",
    "Surprise resignation changed the team dynamic.",
    "Key stakeholder unavailable during critical decision.",
    "Urgent task was completed but not documented.",
    "Software glitch invalidated today's progress.",
    "Anonymous complaint received via external channel.",
    "Ambiguous message caused confusion among peers."
]

SITUATION_BANK = {
    ("preference", "positive"): [
        "I find my best focus comes right after a morning run.",
        "I'm energized by tight deadlines and group brainstorming.",
        "You love starting your day with a strong coffee and a clean inbox.",
        "Collaborative work sessions always lift my mood.",
        "Morning sunlight helps me concentrate for hours.",
        "Brainstorming sessions spark creative solutions.",
        *RARE_CASES
    ],
    ("preference", "negative"): [
        "I get anxious if plans change suddenly, even if it's minor.",
        "Working under bright lights makes it hard for me to concentrate.",
        "Loud environments really drain my productivity.",
        "I dislike meetings that run over their scheduled time.",
        "Excessive notifications interrupt my flow.",
        "Task-switching stresses me out.",
        *RARE_CASES
    ],
    ("preference", "mixed"): [
        "You prefer working independently, but sometimes miss team feedback.",
        "You enjoy flexible schedules, but struggle with self-discipline.",
        "Deadlines motivate me, but I resent the pressure.",
        "You like learning new tools, though it often disrupts your routine.",
        "Flexible hours help, but can make focus harder.",
        "Peer review is useful, yet slows my momentum.",
        *RARE_CASES
    ],
    ("preference", "uncertain"): [
        "I'm not sure if I prefer remote or in-person work environments.",
        "Some days, collaboration helps, other days it's distracting.",
        "You feel torn between structure and spontaneity.",
        "Your feelings about public recognition are mixed and context-dependent.",
        "Sometimes routine is comforting, other times it's limiting.",
        "I'm unsure if silence in meetings signals thoughtfulness or disengagement.",
        *RARE_CASES
    ],
    ("fact", "positive"): [
        "I delivered last quarter's project ahead of schedule.",
        "Your suggestions for the new CRM system improved user adoption rates.",
        "My last presentation received praise from leadership.",
        "The team exceeded its sales targets by 20% this year.",
        "All milestones were hit in the last sprint.",
        "Training sessions resulted in higher test scores.",
        *RARE_CASES
    ],
    ("fact", "negative"): [
        "I missed three deadlines last month due to poor planning.",
        "Team turnover has increased for two consecutive quarters.",
        "Several customer complaints went unresolved for over a week.",
        "Budget tracking errors led to project overruns.",
        "A key deliverable was not submitted on time.",
        "Support tickets have been piling up.",
        *RARE_CASES
    ],
    ("fact", "mixed"): [
        "You consistently finish assigned tasks but rarely go beyond scope.",
        "I make rapid progress on some projects but stall on others.",
        "Our feedback scores are improving, but engagement is dropping.",
        "The product launch was on time, but user adoption is slow.",
        "Quarterly goals were met, yet employee satisfaction dipped.",
        "Implementation is efficient, but documentation lags.",
        *RARE_CASES
    ],
    ("fact", "uncertain"): [
        "It's unclear if the new workflow actually saves time.",
        "Customer satisfaction is steady, but the reasons are ambiguous.",
        "Team morale seems stable, but some feedback contradicts this.",
        "There's conflicting data on the impact of remote work.",
        "We lack enough data to assess the new system's success.",
        "Survey responses are inconclusive.",
        *RARE_CASES
    ],
    ("observation", "positive"): [
        "You seemed motivated and energized during today's team call.",
        "Yesterday, I noticed you voluntarily took on extra work.",
        "I've been reliably consistent with my daily checklists.",
        "Your feedback was constructive and well-received.",
        "You actively encouraged team discussion.",
        "I observed steady improvement in your time management.",
        *RARE_CASES
    ],
    ("observation", "negative"): [
        "Your focus dropped sharply in the afternoon sessions this week.",
        "I caught myself avoiding difficult conversations.",
        "Lately, I’ve noticed frequent lateness to meetings.",
        "You appear overwhelmed by your current workload.",
        "Responses in emails have slowed significantly.",
        "You seem withdrawn during team updates.",
        *RARE_CASES
    ],
    ("observation", "mixed"): [
        "You contributed valuable ideas but interrupted others at times.",
        "I've noticed your work output is strong but occasionally unfocused.",
        "Your enthusiasm is high, but follow-through varies.",
        "Some days, you're proactive, but on others, you seem disengaged.",
        "You offer creative input, but sometimes resist feedback.",
        "Initiative is shown, but priorities shift often.",
        *RARE_CASES
    ],
    ("observation", "uncertain"): [
        "I'm not sure whether your silence in meetings is thoughtful or disengaged.",
        "Your mood changes are subtle and hard to interpret.",
        "Team morale appears fine, but there's occasional tension.",
        "It's unclear if recent changes affected productivity.",
        "Your participation level fluctuates unpredictably.",
        "It's not obvious if you benefit from feedback sessions.",
        *RARE_CASES
    ],
}

EDGE_CASE_BANK = [
    "This situation contains contradictory feedback and cannot be easily categorized.",
    "I have no recollection of this event but found it in my logs.",
    "The context for this memory is missing; unsure if it should be stored.",
    "There's insufficient evidence to determine the accuracy of this observation.",
    "Two team members gave opposite reports about the same event.",
    "This entry is a duplicate of an earlier, but with conflicting scores.",
    "Memory appears emotionally charged but lacks factual details.",
    "Entry contains both positive and negative signals simultaneously.",
    "This situation might not be relevant to long-term decision-making.",
    "There is uncertainty if this memory pertains to the current team or a previous one.",
    "This is an outdated observation and may not reflect the current context.",
    "The reasoning for this memory is ambiguous or inconclusive.",
    "This detail is highly specific but unlikely to impact future actions.",
    "Assessment is complicated by inconsistent source data.",
    "The significance of this event is unclear.",
    "Multiple parties disagree on the facts.",
    "Input format does not match any known pattern.",
    "Partial information only; key details missing.",
    "Simultaneous success and failure reported for same event.",
    "Personal notes conflict with official record.",
    "Outcome changed after this was first logged."
]

REASONING_TEMPLATES = [
    "Given the details, storing this might provide useful historical context.",
    "This entry stands out due to its unusual combination of factors.",
    "Choosing '{choice}' may impact future group outcomes.",
    "The reported facts seem inconsistent, so further review is suggested.",
    "Emotional aspects are prominent, but evidence is mixed.",
    "Contextual clues in '{situation}' suggest ambiguity.",
    "Patterns like this have previously led to unexpected outcomes.",
    "Multiple interpretations are possible—flag for deeper analysis.",
    "This event aligns with observed trends, but relevance is unclear.",
    "This is a rare or contradictory scenario, so caution is advised.",
    "Based on team experience, uncertainty here could signal bigger issues.",
    "This memory includes contradictory elements; decide conservatively.",
    "No clear precedent exists; best practice is to retain for now.",
    "Edge/contradictory scenario: Requires leadership input.",
    "Judgment call needed—data is incomplete."
]

def generate_reasoning(situation, t_type, sentiment, choice, edge_case=False):
    templ = random.choice(REASONING_TEMPLATES)
    r = templ.format(situation=situation[:60], choice=choice, type=t_type, sentiment=sentiment)
    if random.random() < 0.35:
        r += f" (Situation summary: {situation[:36]}...)"
    if edge_case and "Edge/contradictory scenario" not in r:
        r = "Edge/contradictory scenario: " + r
    return r

def ensure_fields(entry, types, sentiments):
    if not entry.get("type"):
        entry["type"] = random.choice(types)
    if not entry.get("sentiment"):
        entry["sentiment"] = random.choice(sentiments)
    return entry

def generate_decision_sample(existing_situations, types, sentiments, t_type, sentiment, edge_case=False):
    use_rare = (not edge_case) and random.randint(1,75) == 1
    if edge_case:
        situation = random.choice(EDGE_CASE_BANK)
    elif use_rare:
        situation = random.choice(RARE_CASES)
    else:
        key = (t_type, sentiment)
        bank = SITUATION_BANK.get(key, [])
        situation = random.choice(bank) if bank else "General scenario."
    tries = 0
    while situation.lower() in existing_situations:
        tries += 1
        if edge_case:
            situation = random.choice(EDGE_CASE_BANK)
        elif use_rare:
            situation = random.choice(RARE_CASES)
        else:
            key = (t_type, sentiment)
            situation = random.choice(SITUATION_BANK.get(key, ["General scenario."]))
        if tries > 9:
            break
    existing_situations.add(situation.lower())
    options = DECISION_OPTIONS.copy()
    if random.random() < 0.22:
        options += random.choice(SCENARIO_SPECIFIC_OPTIONS)
        options = list(sorted(set(options)))
    if edge_case or sentiment in ("uncertain", "mixed") or random.random() < 0.08:
        if "Flag for further review" not in options:
            options.append("Flag for further review")
    choice_probs = {
        "Store this in long-term memory": 0.57 if sentiment in ("positive",) else 0.45,
        "Discard it for now": 0.32 if sentiment == "negative" else 0.45,
        "Flag for further review": 0.11 if edge_case or sentiment in ("mixed","uncertain") else 0.10
    }
    choice = random.choices(list(choice_probs), weights=list(choice_probs.values()), k=1)[0]
    if edge_case:
        choice = random.choice(options)
    reasoning = generate_reasoning(situation, t_type, sentiment, choice, edge_case)
    confidence = round(random.uniform(0.45, 0.99 if choice == "Store this in long-term memory" else 0.87), 2)
    if edge_case:
        confidence = round(random.uniform(0.2, 0.75), 2)
    importance_score = random.randint(1, 7 if choice != "Discard it for now" else 3)
    entry = {
        "situation": situation,
        "options": options,
        "choice": choice,
        "reasoning": reasoning,
        "confidence": confidence,
        "type": t_type,
        "sentiment": sentiment,
        "importance_score": importance_score
    }
    return ensure_fields(entry, types, sentiments)

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def expand_and_fill(data, target_size=100_000, edge_case_ratio=0.17):
    existing_situations = set(x["situation"].strip().lower() for x in data if x.get("situation"))
    print(f"🟦 Starting with {len(data)} entries, {len(existing_situations)} unique situations.")
    needed = target_size - len(data)
    new_entries = []
    types = ["preference", "fact", "observation"]
    sentiments = ["positive", "negative", "mixed", "uncertain"]
    edge_cases_needed = int(needed * edge_case_ratio)
    print(f"⚡ Adding {edge_cases_needed} edge/contradictory/ambiguous cases for legendary diversity.")
    for _ in range(edge_cases_needed):
        t_type = random.choice(types)
        sentiment = random.choice(sentiments)
        entry = generate_decision_sample(existing_situations, types, sentiments, t_type, sentiment, edge_case=True)
        new_entries.append(entry)
    for _ in range(needed - edge_cases_needed):
        t_type = random.choices(types, weights=[0.4, 0.35, 0.25])[0]
        sentiment = random.choices(sentiments, weights=[0.45, 0.25, 0.20, 0.10])[0]
        entry = generate_decision_sample(existing_situations, types, sentiments, t_type, sentiment, edge_case=False)
        new_entries.append(entry)
    print(f"✅ Created {len(new_entries)} truly unique and scenario-diverse entries.")
    expanded = data + new_entries
    random.shuffle(expanded)
    print(f"✅ Final legendary dataset size: {len(expanded)}")
    return expanded

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input cleaned decision JSONL")
    parser.add_argument("output", help="Path to save expanded JSONL")
    parser.add_argument("--target", type=int, default=100000, help="Target dataset size (default 100k)")
    args = parser.parse_args()
    raw = load_jsonl(args.input)
    expanded = expand_and_fill(raw, target_size=args.target)
    save_jsonl(expanded, args.output)
    print(f"✨ Done. Saved to: {args.output}")
