"""
Step 2: Auto-identify speakers from the diarized transcript.

Heuristic approach:
  - Looks for patterns like "Thank you, [Name]", "Over to you, [Name]",
    "[Name], your response", moderator introductions, etc.
  - Produces a speaker_map.json you should review and correct before Step 3.
  - Prints a cross-file reconciliation report so you can see which SPEAKER_XX
    labels across different recordings are probably the same real person.

Usage:
    python src/identify_speakers.py --transcripts data/transcripts/

Outputs:
    data/speaker_map.json   <-- REVIEW AND EDIT THIS FILE before running generate_quiz.py

Key concept: speaker IDs (SPEAKER_00, SPEAKER_01, ...) are assigned fresh for
each audio file, so the same real person will get different IDs across recordings.
The speaker_map.json uses the candidate's real NAME as the join key — if you set
two different speaker IDs (from different files) to the same name, their utterances
will be automatically merged when generate_quiz.py builds the corpus.
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
    "Morning", "Evening", "Afternoon", "Welcome", "Hello", "Tonight", "Yeah", "Again", "Look", "Sorry", "Time", "Second", "Judgingly", "Secondly"
}

# Minimum vote-score to treat a name guess as "confident" in the reconciliation report
CONFIDENT_VOTE_THRESHOLD = 2


def load_transcripts(transcript_dir: str) -> list[dict]:
    """Load all diarized transcript JSONs from a directory, tagging each with its date."""
    transcript_dir = Path(transcript_dir)
    transcripts = []
    for f in sorted(transcript_dir.glob("*_diarized.json")):
        with open(f) as fp:
            data = json.load(fp)
            data["_source_file"] = str(f)
            # Extract date from filename like 2025-09-10_diarized.json
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
            data["_date"] = date_match.group(1) if date_match else f.stem
            transcripts.append(data)
    return transcripts


def extract_name_votes_per_file(
    transcripts: list[dict],
) -> dict[str, dict[str, dict[str, int]]]:
    """
    Run name-extraction heuristics on each transcript file independently.

    Returns:
        {
          "2025-09-10": { "SPEAKER_00": {"Alice Johnson": 3, ...}, ... },
          "2025-09-27": { "SPEAKER_02": {"Alice Johnson": 2, ...}, ... },
          ...
        }
    """
    per_file_votes: dict[str, dict[str, dict[str, int]]] = {}

    for transcript in transcripts:
        date = transcript["_date"]
        segments = transcript.get("segments", [])
        speaker_name_votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "")

            for pattern in INTRODUCTION_PATTERNS:
                for match in re.finditer(pattern, text):
                    name = match.group(1).strip()
                    if name in NAME_BLACKLIST or len(name) < 3:
                        continue

                    # "over to you, Alice" → next speaker is probably Alice
                    if any(kw in text.lower() for kw in ("over to you", "your turn", "next")):
                        if i + 1 < len(segments):
                            next_speaker = segments[i + 1].get("speaker", "")
                            if next_speaker and next_speaker != speaker:
                                speaker_name_votes[next_speaker][name] += 2

                    # "I'm Alice Smith" → this speaker IS Alice
                    if re.search(r"(?:I'?m|[Mm]y name is)\s+", text):
                        if speaker:
                            speaker_name_votes[speaker][name] += 3

                    # Generic proximity signal
                    if speaker:
                        speaker_name_votes[speaker][name] += 1

        per_file_votes[date] = {
            spk: dict(votes) for spk, votes in speaker_name_votes.items()
        }

    return per_file_votes


def build_per_file_stats(transcripts: list[dict]) -> dict[str, dict[str, dict]]:
    """
    Compute word/segment counts per speaker, broken out by file.

    Returns:
        { "2025-09-10": { "SPEAKER_00": {"words": 1240, "segments": 42}, ... }, ... }
    """
    stats: dict[str, dict[str, dict]] = {}
    for transcript in transcripts:
        date = transcript["_date"]
        file_stats: dict[str, dict] = defaultdict(lambda: {"words": 0, "segments": 0})
        for seg in transcript.get("segments", []):
            spk = seg.get("speaker", "UNKNOWN")
            file_stats[spk]["words"] += len(seg.get("text", "").split())
            file_stats[spk]["segments"] += 1
        stats[date] = dict(file_stats)
    return stats


def build_speaker_map(
    transcripts: list[dict],
    per_file_votes: dict[str, dict[str, dict[str, int]]],
    per_file_stats: dict[str, dict[str, dict]],
) -> dict:
    """
    Build the flat speaker_map.json.

    Keys are scoped as "DATE/SPEAKER_XX" to avoid collisions across files
    (e.g. "2025-09-10/SPEAKER_00" and "2025-09-27/SPEAKER_00" are different people).
    """
    speaker_map = {}

    for date, file_votes in sorted(per_file_votes.items()):
        file_stats = per_file_stats.get(date, {})

        # Also include speakers with stats but no name votes
        all_speakers_in_file = set(file_stats.keys()) | set(file_votes.keys())

        for spk in sorted(all_speakers_in_file):
            key = f"{date}/{spk}"
            votes = file_votes.get(spk, {})
            best_guess = max(votes, key=votes.get) if votes else None
            stats = file_stats.get(spk, {"words": 0, "segments": 0})

            speaker_map[key] = {
                "name": best_guess,       # <-- EDIT THIS: real candidate name (shared across files)
                "role": "candidate",      # <-- EDIT THIS: "candidate" or "moderator"
                "source_file": date,
                "speaker_id": spk,
                "name_votes": votes,
                "word_count": stats["words"],
                "segment_count": stats["segments"],
                "include_in_quiz": True,  # <-- set False for moderators/non-candidates
            }

    return speaker_map


def print_reconciliation_report(
    per_file_votes: dict[str, dict[str, dict[str, int]]],
    per_file_stats: dict[str, dict[str, dict]],
    dates: list[str],
) -> None:
    """
    Print a cross-file reconciliation report grouping speaker IDs by their
    best-guess name across files.

    For each name that appears in ≥2 files, shows which SPEAKER_XX to map to
    that name in each file so you can verify before editing speaker_map.json.

    For speakers with no name evidence, lists them separately for manual lookup.
    """
    # Build: name → { date → [(speaker_id, vote_score, word_count)] }
    name_to_file_speakers: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    unidentified: dict[str, list] = defaultdict(list)  # date → [speaker_ids]

    for date in dates:
        file_votes = per_file_votes.get(date, {})
        file_stats = per_file_stats.get(date, {})
        all_speakers = set(file_stats.keys()) | set(file_votes.keys())

        for spk in sorted(all_speakers):
            votes = file_votes.get(spk, {})
            words = file_stats.get(spk, {}).get("words", 0)

            if votes:
                best_name = max(votes, key=votes.get)
                best_score = votes[best_name]
                name_to_file_speakers[best_name][date].append((spk, best_score, words))
            else:
                unidentified[date].append((spk, words))

    print("\n" + "=" * 70)
    print("CROSS-FILE SPEAKER RECONCILIATION REPORT")
    print("=" * 70)
    print(
        "Each block below shows a name that was detected in one or more files.\n"
        "Speakers grouped under the same name should be set to that name\n"
        "in speaker_map.json so their utterances are merged.\n"
    )

    # Sort: names appearing in most files first, then alphabetically
    sorted_names = sorted(
        name_to_file_speakers.items(),
        key=lambda x: (-len(x[1]), x[0]),
    )

    confirmed_cross_file = []
    single_file_names = []

    for name, date_map in sorted_names:
        if len(date_map) >= 2:
            confirmed_cross_file.append((name, date_map))
        else:
            single_file_names.append((name, date_map))

    # ── Cross-file matches ──────────────────────────────────────────────────
    if confirmed_cross_file:
        print("── Likely same person across multiple files ──────────────────────────\n")
        for name, date_map in confirmed_cross_file:
            confident = all(
                any(score >= CONFIDENT_VOTE_THRESHOLD for _, score, _ in speakers)
                for speakers in date_map.values()
            )
            confidence_tag = "✓ confident" if confident else "? low confidence"
            print(f'  "{name}"  [{confidence_tag}]')
            for date in dates:
                if date in date_map:
                    for spk, score, words in date_map[date]:
                        print(f"    {date}  →  {spk:15s}  (votes: {score}, words: {words:,})")
            print(f'    → Set all of the above to  "name": "{name}"  in speaker_map.json')
            print()
    else:
        print("  (No names were detected in more than one file yet.)\n")

    # ── Single-file detections ──────────────────────────────────────────────
    if single_file_names:
        print("── Detected in only one file (may be new candidate or low evidence) ──\n")
        for name, date_map in single_file_names:
            for date, speakers in date_map.items():
                for spk, score, words in speakers:
                    print(
                        f"  {date}  {spk:15s}  →  \"{name}\"  "
                        f"(votes: {score}, words: {words:,})"
                    )
        print()

    # ── Unidentified speakers ───────────────────────────────────────────────
    any_unidentified = any(entries for entries in unidentified.values())
    if any_unidentified:
        print("── No name detected — manual identification needed ───────────────────\n")
        for date in dates:
            for spk, words in unidentified.get(date, []):
                print(f"  {date}  {spk:15s}  words: {words:,}  →  name: ???")
        print(
            "\n  Tip: word_count helps spot the moderator (usually highest among non-candidates)."
            "\n  Open the transcript JSON and search for segments from this speaker_id to identify them."
        )
        print()

    print("=" * 70)
    print("ACTION: Edit data/speaker_map.json so that the same real person has")
    print("        the same 'name' value in every file they appear in.")
    print("        Set 'include_in_quiz': false for moderators and non-candidates.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Auto-identify speakers and produce a speaker_map.json for human review. "
            "Also prints a cross-file reconciliation report to help you merge the same "
            "speaker across multiple recordings."
        )
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

    dates = [t["_date"] for t in transcripts]
    print(f"Loaded {len(transcripts)} transcript(s): {', '.join(dates)}")

    per_file_votes = extract_name_votes_per_file(transcripts)
    per_file_stats = build_per_file_stats(transcripts)
    speaker_map = build_speaker_map(transcripts, per_file_votes, per_file_stats)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(speaker_map, f, indent=2)

    # ── Per-file summary table ──────────────────────────────────────────────
    print(f"\n=== SPEAKER SUMMARY (by file) ===\n")
    for date in dates:
        print(f"  {date}:")
        file_entries = {
            k: v for k, v in speaker_map.items() if v["source_file"] == date
        }
        for key, info in sorted(file_entries.items(), key=lambda x: x[1]["word_count"], reverse=True):
            guess = info["name"] or "UNKNOWN"
            print(
                f"    {info['speaker_id']:15s}  →  {guess:25s}  "
                f"words: {info['word_count']:,}"
            )
        print()

    # ── Cross-file reconciliation report ───────────────────────────────────
    print_reconciliation_report(per_file_votes, per_file_stats, dates)

    print(f"Saved: {output_path}")
    print(f"Keys use 'DATE/SPEAKER_XX' format to avoid collisions across files.")
    print(f"generate_quiz.py joins by 'name', so set the same name for the same person.\n")


if __name__ == "__main__":
    main()
