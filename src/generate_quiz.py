"""
Step 3: Generate quiz questions grounded in transcript quotes.

Only emits a question if the LLM can provide a verbatim supporting quote
from the transcript for at least one candidate's position. This directly
addresses the defamation-risk concern.

Usage:
    python src/generate_quiz.py

Requires:
    - data/transcripts/*_diarized.json  (from Step 1)
    - data/speaker_map.json             (from Step 2, after your edits)
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

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError("ANTHROPIC_API_KEY not set in .env file.")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Likert scale answer options (same for every question)
LIKERT_OPTIONS = [
    {"id": "A", "text": "Strongly agree"},
    {"id": "B", "text": "Somewhat agree"},
    {"id": "C", "text": "Somewhat disagree"},
    {"id": "D", "text": "Strongly disagree"},
]

TARGET_QUESTIONS = 10  # aim for this many questions


def load_transcripts(transcript_dir: str) -> list[dict]:
    transcript_dir = Path(transcript_dir)
    transcripts = []
    for f in sorted(transcript_dir.glob("*_diarized.json")):
        with open(f) as fp:
            transcripts.append(json.load(fp))
    return transcripts


def load_speaker_map(speaker_map_path: str) -> dict:
    with open(speaker_map_path) as f:
        return json.load(f)


def build_candidate_corpora(transcripts: list[dict], speaker_map: dict) -> dict[str, list[str]]:
    """
    Build a per-candidate corpus of their spoken text.
    Returns: {candidate_name: [utterances...]}
    """
    # speaker_id → candidate name (only for included candidates)
    id_to_name = {}
    for speaker_id, info in speaker_map.items():
        if info.get("include_in_quiz") and info.get("role") == "candidate" and info.get("name"):
            id_to_name[speaker_id] = info["name"]

    corpora = {name: [] for name in id_to_name.values()}

    for transcript in transcripts:
        for seg in transcript.get("segments", []):
            speaker = seg.get("speaker", "")
            text = seg.get("text", "").strip()
            if speaker in id_to_name and text:
                corpora[id_to_name[speaker]].append(text)

    return corpora


def build_full_corpus_text(corpora: dict[str, list[str]]) -> str:
    """Format the candidate corpora into a labeled text block for the LLM."""
    parts = []
    for candidate, utterances in corpora.items():
        combined = " ".join(utterances)
        # Truncate very long corpora to stay within context limits
        if len(combined) > 15000:
            combined = combined[:15000] + "... [truncated]"
        parts.append(f"=== {candidate} ===\n{combined}")
    return "\n\n".join(parts)


def generate_questions_and_positions(corpus_text: str, candidates: list[str]) -> list[dict]:
    """
    Ask Claude to:
    1. Identify key recurring themes/issues from the transcripts
    2. For each theme, draft a quiz question (agree/disagree statement)
    3. For each candidate × question, predict their Likert response AND provide
       a verbatim quote (≤30 words) from their corpus supporting the prediction.
    4. If no supporting quote exists, omit that candidate from that question.

    Returns a list of raw question dicts (to be validated below).
    """
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
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)["questions"]


def validate_and_filter_questions(raw_questions: list[dict]) -> list[dict]:
    """
    Filter out:
    - Questions where no candidate has a non-null quote
    - Candidates with null quotes (drop them from that question)
    - Questions where fewer than 2 candidates have quotes
    """
    valid_questions = []

    for q in raw_questions:
        positions = q.get("candidate_positions", {})

        # Keep only candidates with actual quotes
        grounded = {
            name: pos
            for name, pos in positions.items()
            if pos.get("quote") and pos["quote"] != "null"
        }

        if len(grounded) < 2:
            print(f"  Skipping '{q['topic']}': only {len(grounded)} candidate(s) have quotes")
            continue

        # Validate Likert answers
        for name, pos in grounded.items():
            if pos.get("answer") not in {"A", "B", "C", "D"}:
                print(f"  Warning: invalid answer '{pos.get('answer')}' for {name} on '{q['topic']}'")
                pos["answer"] = "B"  # fallback

        valid_questions.append({
            "topic": q["topic"],
            "question": q["question"],
            "options": LIKERT_OPTIONS,
            "candidate_positions": grounded,
        })

    return valid_questions


def build_quiz_data(questions: list[dict], speaker_map: dict) -> dict:
    """
    Build the final quiz_data.json structure consumed by the frontend.
    """
    # Build candidate list with metadata
    candidates = []
    for speaker_id, info in speaker_map.items():
        if info.get("include_in_quiz") and info.get("role") == "candidate" and info.get("name"):
            candidates.append({
                "name": info["name"],
                "speaker_id": speaker_id,
            })

    # Deduplicate (multiple speaker IDs could map to same name — shouldn't happen but safety)
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
    transcript_dir = "data/transcripts"
    speaker_map_path = "data/speaker_map.json"
    output_path = "data/quiz_data.json"

    print("[1/5] Loading transcripts...")
    transcripts = load_transcripts(transcript_dir)
    if not transcripts:
        print(f"ERROR: No transcripts found in {transcript_dir}. Run transcribe.py first.")
        return
    print(f"      Loaded {len(transcripts)} transcript(s)")

    print("[2/5] Loading speaker map...")
    if not Path(speaker_map_path).exists():
        print(f"ERROR: {speaker_map_path} not found. Run identify_speakers.py first.")
        return
    speaker_map = load_speaker_map(speaker_map_path)

    # Validate speaker map has at least some candidates
    candidates = [
        info["name"]
        for info in speaker_map.values()
        if info.get("include_in_quiz") and info.get("role") == "candidate" and info.get("name")
    ]
    if len(candidates) < 2:
        print(f"ERROR: Need at least 2 candidates in speaker_map.json with include_in_quiz=true.")
        print(f"       Found: {candidates}")
        print(f"       Edit {speaker_map_path} and try again.")
        return
    print(f"      Candidates: {candidates}")

    print("[3/5] Building candidate corpora...")
    corpora = build_candidate_corpora(transcripts, speaker_map)
    corpus_text = build_full_corpus_text(corpora)
    total_words = sum(len(u.split()) for utterances in corpora.values() for u in utterances)
    print(f"      Total words across all candidates: {total_words:,}")

    print(f"[4/5] Generating quiz questions (targeting {TARGET_QUESTIONS})...")
    raw_questions = generate_questions_and_positions(corpus_text, candidates)
    print(f"      Claude returned {len(raw_questions)} candidate questions")

    print("[5/5] Validating and filtering questions...")
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
