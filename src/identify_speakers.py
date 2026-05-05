"""
Step 2: Auto-identify speakers from the diarized transcript.

Heuristic approach:
  - Looks for patterns like "Thank you, [Name]", "Over to you, [Name]",
    "[Name], your response", moderator introductions, etc.
  - Produces a speaker_map.json you should review and correct before Step 3.

Usage:
    python src/identify_speakers.py --transcripts data/transcripts/

Outputs:
    data/speaker_map.json   <-- REVIEW AND EDIT THIS FILE before running generate_quiz.py
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


# Patterns that suggest the speaker just identified is named NAME
# These fire when someone (usually a moderator) addresses or introduces a candidate by name
INTRODUCTION_PATTERNS = [
    # Moderator calls on someone: "Over to you, Alex", "Your turn, Sarah"
    r"(?:over to you|your turn|next (?:question )?(?:goes )?to|thank you[,.]?\s+)[\s,]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    # Introducing themselves: "I'm Alex Smith" / "My name is Alex Smith"
    r"(?:I'?m|[Mm]y name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
    # Moderator says "Candidate Smith, you have..." 
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s+you have\s+(?:one|two|three|\d+)\s+minute",
    # Direct address at start of segment: "Sarah, ..."
    r"^([A-Z][a-z]+),\s",
]

# Words that look like names but are definitely not candidate names
NAME_BLACKLIST = {
    "Thank", "The", "This", "That", "So", "And", "But", "Now", "Yes", "No",
    "Well", "Please", "Next", "Over", "Your", "My", "Our", "Their", "Its",
    "Cambridge", "School", "Board", "Council", "City", "Community", "Good",
    "Morning", "Evening", "Afternoon", "Welcome", "Hello", "Tonight",
}


def load_transcripts(transcript_dir: str) -> list[dict]:
    """Load all diarized transcript JSONs from a directory."""
    transcript_dir = Path(transcript_dir)
    transcripts = []
    for f in sorted(transcript_dir.glob("*_diarized.json")):
        with open(f) as fp:
            data = json.load(fp)
            data["_source_file"] = str(f)
            transcripts.append(data)
    return transcripts


def extract_name_candidates(segments: list[dict]) -> dict[str, dict[str, int]]:
    """
    For each SPEAKER_XX, collect candidate real names based on pattern matching.
    Returns: {speaker_id: {candidate_name: count}}
    """
    # First pass: find which speaker IDs are immediately followed/preceded by name mentions
    # We look at segments in context: if segment N mentions "Thank you, Alice"
    # and segment N+1 is from SPEAKER_03, that's evidence SPEAKER_03 is Alice.

    speaker_name_votes = defaultdict(lambda: defaultdict(int))

    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "")

        for pattern in INTRODUCTION_PATTERNS:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if name in NAME_BLACKLIST or len(name) < 3:
                    continue

                # If a moderator says "over to you, Alice", the NEXT segment's speaker is probably Alice
                if "over to you" in text.lower() or "your turn" in text.lower() or "next" in text.lower():
                    if i + 1 < len(segments):
                        next_speaker = segments[i + 1].get("speaker", "")
                        if next_speaker and next_speaker != speaker:
                            speaker_name_votes[next_speaker][name] += 2  # stronger signal

                # If someone says "I'm Alice Smith", that speaker IS Alice
                if re.search(r"(?:I'?m|[Mm]y name is)\s+", text):
                    if speaker:
                        speaker_name_votes[speaker][name] += 3  # strongest signal

                # Generic: mentioned near this speaker
                if speaker:
                    speaker_name_votes[speaker][name] += 1

    return {k: dict(v) for k, v in speaker_name_votes.items()}


def build_speaker_map(transcripts: list[dict]) -> dict:
    """
    Build a speaker_map by analyzing all transcripts.
    Returns a dict ready to be saved as speaker_map.json.
    """
    all_segments = []
    for t in transcripts:
        all_segments.extend(t.get("segments", []))

    name_votes = extract_name_candidates(all_segments)

    # Also collect basic stats: word count per speaker (helps identify moderator as highest-speaking)
    word_counts = defaultdict(int)
    segment_counts = defaultdict(int)
    for seg in all_segments:
        speaker = seg.get("speaker", "UNKNOWN")
        words = len(seg.get("text", "").split())
        word_counts[speaker] += words
        segment_counts[speaker] += 1

    all_speakers = set(word_counts.keys())

    speaker_map = {}
    for speaker in sorted(all_speakers):
        votes = name_votes.get(speaker, {})
        best_guess = max(votes, key=votes.get) if votes else None
        speaker_map[speaker] = {
            "name": best_guess,          # <-- EDIT THIS: set to the candidate's real name
            "role": "candidate",         # <-- EDIT THIS: "candidate" or "moderator"
            "name_votes": votes,         # evidence for the auto-guess (for your reference)
            "word_count": word_counts[speaker],
            "segment_count": segment_counts[speaker],
            "include_in_quiz": True,     # <-- set to False for moderators/non-candidates
        }

    return speaker_map


def main():
    parser = argparse.ArgumentParser(
        description="Auto-identify speakers and produce a speaker_map.json for human review."
    )
    parser.add_argument(
        "--transcripts",
        default="data/transcripts",
        help="Directory containing *_diarized.json files",
    )
    parser.add_argument(
        "--output",
        default="data/speaker_map.json",
        help="Where to write speaker_map.json",
    )
    args = parser.parse_args()

    transcripts = load_transcripts(args.transcripts)
    if not transcripts:
        print(f"No *_diarized.json files found in {args.transcripts}")
        return

    print(f"Loaded {len(transcripts)} transcript(s) with segments from all dates.")

    speaker_map = build_speaker_map(transcripts)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(speaker_map, f, indent=2)

    print(f"\nSaved speaker map to: {output_path}")
    print("\n=== SPEAKER SUMMARY ===")
    print("(Review and edit this file before running generate_quiz.py)\n")
    for speaker_id, info in speaker_map.items():
        guess = info["name"] or "UNKNOWN"
        role = info["role"]
        words = info["word_count"]
        print(f"  {speaker_id:15s} → name: {guess:25s} | role: {role} | words: {words:,}")

    print(f"\nEdit {output_path} to correct any wrong names or roles.")
    print("Set 'include_in_quiz': false for moderators and non-candidates.")


if __name__ == "__main__":
    main()
