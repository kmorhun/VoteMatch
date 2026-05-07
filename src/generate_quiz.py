"""
Step 3b: Generate quiz questions grounded in transcript quotes.

Only emits a question if Claude can provide a verbatim supporting quote
from the transcript for at least two candidates' positions.

Usage:
    python src/generate_quiz.py

Requires:
    - data/candidate_corpora.json  (from build_corpora.py)
    - data/speaker_map.json        (from identify_speakers.py)
    - ANTHROPIC_API_KEY in .env

Outputs:
    data/quiz_data.json   (consumed by the GitHub Pages site)
"""

import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from typing import List, Dict

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError("ANTHROPIC_API_KEY not set in .env file.")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

LIKERT_OPTIONS = [
    {"id": "A", "text": "Strongly agree"},
    {"id": "B", "text": "Somewhat agree"},
    {"id": "C", "text": "Somewhat disagree"},
    {"id": "D", "text": "Strongly disagree"},
]

TARGET_QUESTIONS = 10


def load_corpora(corpora_path: str) -> Dict:
    with open(corpora_path) as f:
        return json.load(f)


def load_speaker_map(speaker_map_path: str) -> dict:
    with open(speaker_map_path) as f:
        return json.load(f)


def build_full_corpus_text(corpora: Dict[str, List[str]]) -> str:
    parts = []
    for candidate, utterances in corpora.items():
        combined = " ".join(utterances)
        if len(combined) > 5000:
            combined = combined[:5000] + "... [truncated]"
        parts.append(f"=== {candidate} ===\n{combined}")
    return "\n\n".join(parts)


def generate_questions_and_positions(corpus_text: str, candidates: List[str]) -> List[Dict]:
    candidate_list = ", ".join(candidates)

    prompt = f"""You are analyzing transcripts from a Cambridge, MA School Committee candidate forum.
Your job is to generate quiz questions for a voter-matching tool.

CRITICAL RULE: For each question, you must provide a verbatim quote from the candidate's
transcript to justify their predicted position. If you cannot find a direct quote that
supports a position, you must omit that candidate from that question entirely.
Do NOT infer or extrapolate — only use what was explicitly said.

Here are the candidate transcripts:

{corpus_text}

---

TASK:
1. Identify {TARGET_QUESTIONS} key policy topics that came up repeatedly across candidates.
   Focus on topics where candidates expressed different views (not topics all agreed on).
   Good examples for school committee: curriculum choices, budget priorities, school hours,
   standardized testing, special education funding, teacher evaluation, school start times, etc.

2. For each topic, write an agree/disagree statement (the quiz question).
   The statement should be clear, concrete, and position-taking — NOT wishy-washy.
   Example: "The district should offer Algebra I to 8th graders" (not "Math education is important")

3. For each candidate, predict their Likert response:
   A = Strongly agree, B = Somewhat agree, C = Somewhat disagree, D = Strongly disagree

4. For each prediction, provide a verbatim quote (under 30 words) from the candidate's
   transcript. If no clear supporting quote exists, write null for that candidate.

Respond ONLY with valid JSON in exactly this format:
{{
  "questions": [
    {{
      "topic": "Brief topic label (e.g., 'Algebra I in 8th grade')",
      "question": "The full agree/disagree statement shown to voters",
      "candidate_positions": {{
        "Candidate Name": {{
          "answer": "A",
          "quote": "Exact verbatim quote from their transcript supporting this position"
        }},
        "Another Candidate": {{
          "answer": "C",
          "quote": null
        }}
      }}
    }}
  ]
}}

Candidates to include: {candidate_list}
Return JSON only, no other text."""

    print("  Calling Claude API to generate quiz questions...")
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)["questions"]
    except json.JSONDecodeError as e:
        print(f"\nERROR: Failed to parse Claude's response as JSON: {e}")
        print(f"Stop reason: {message.stop_reason}")
        print(f"Raw response (last 500 chars):\n{raw[-500:]}")
        raise


def validate_and_filter_questions(raw_questions: List[Dict]) -> List[Dict]:
    valid_questions = []

    for q in raw_questions:
        positions = q.get("candidate_positions", {})

        grounded = {
            name: pos
            for name, pos in positions.items()
            if pos.get("quote") and pos["quote"] != "null"
        }

        if len(grounded) < 2:
            print(f"  Skipping '{q['topic']}': only {len(grounded)} candidate(s) have quotes")
            continue

        for name, pos in grounded.items():
            if pos.get("answer") not in {"A", "B", "C", "D"}:
                print(f"  Warning: invalid answer '{pos.get('answer')}' for {name} on '{q['topic']}'")
                pos["answer"] = "B"

        valid_questions.append({
            "topic": q["topic"],
            "question": q["question"],
            "options": LIKERT_OPTIONS,
            "candidate_positions": grounded,
        })

    return valid_questions


def build_quiz_data(questions: List[Dict], speaker_map: Dict) -> Dict:
    candidates = []
    for speaker_id, info in speaker_map.items():
        if info.get("include_in_quiz") and info.get("role") == "candidate" and info.get("name"):
            candidates.append({"name": info["name"], "speaker_id": speaker_id})

    seen = set()
    unique_candidates = []
    for c in candidates:
        if c["name"] not in seen:
            seen.add(c["name"])
            unique_candidates.append(c)

    return {
        "meta": {
            "election": "Cambridge School Committee 2025",
            "generated_by": "VoteMatch pipeline",
            "note": "Candidate positions are inferred from public forum transcripts. "
                    "Each position is accompanied by a verbatim quote from the candidate. "
                    "This tool is for informational purposes only.",
        },
        "candidates": unique_candidates,
        "questions": questions,
    }


def main():
    import sys
    use_cached = "--cached" in sys.argv

    corpora_path = "data/candidate_corpora.json"
    speaker_map_path = "data/speaker_map.json"
    raw_questions_path = "data/raw_questions.json"
    output_path = "data/quiz_data.json"

    print("[1/4] Loading candidate corpora...")
    if not Path(corpora_path).exists():
        print(f"ERROR: {corpora_path} not found. Run build_corpora.py first.")
        return
    corpora_data = load_corpora(corpora_path)
    corpora = corpora_data["candidates"]
    stats = corpora_data.get("stats", {})
    for name, s in stats.items():
        print(f"      {name}: {s['utterance_count']} utterances, {s['word_count']:,} words")

    print("[2/4] Loading speaker map...")
    speaker_map = load_speaker_map(speaker_map_path)
    candidates = list(corpora.keys())
    print(f"      Candidates: {candidates}")

    if use_cached and Path(raw_questions_path).exists():
        print(f"[3/4] Loading cached questions from {raw_questions_path} (skipping API call)...")
        with open(raw_questions_path) as f:
            raw_questions = json.load(f)
    else:
        corpus_text = build_full_corpus_text(corpora)
        print(f"[3/4] Generating quiz questions (targeting {TARGET_QUESTIONS})...")
        raw_questions = generate_questions_and_positions(corpus_text, candidates)
        with open(raw_questions_path, "w") as f:
            json.dump(raw_questions, f, indent=2)
        print(f"      Saved raw questions to {raw_questions_path}")
    print(f"      {len(raw_questions)} candidate questions")

    print("[4/4] Validating and filtering questions...")
    valid_questions = validate_and_filter_questions(raw_questions)
    print(f"      {len(valid_questions)} questions passed validation (have ≥2 grounded candidates)")

    quiz_data = build_quiz_data(valid_questions, speaker_map)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(quiz_data, f, indent=2)

    print(f"\nDone! Quiz data saved to: {output_path}")
    print(f"Questions: {len(valid_questions)}")
    print(f"Candidates: {[c['name'] for c in quiz_data['candidates']]}")
    print(f"\nNext step: copy {output_path} to docs/quiz_data.json and deploy to GitHub Pages.")


if __name__ == "__main__":
    main()
