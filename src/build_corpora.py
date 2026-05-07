"""
Step 3a: Build per-candidate corpora from diarized transcripts.

Sanity-check the output before running generate_quiz.py.

Usage:
    python src/build_corpora.py

Requires:
    - data/transcripts/*_diarized.json  (from Step 1)
    - data/speaker_map.json             (from Step 2, after your edits)

Outputs:
    data/candidate_corpora.json
"""

import json
from pathlib import Path
from typing import Dict, List


def load_transcripts(transcript_dir: str) -> List[Dict]:
    transcript_dir = Path(transcript_dir)
    transcripts = []
    for f in sorted(transcript_dir.glob("*_diarized.json")):
        with open(f) as fp:
            data = json.load(fp)
        # Inject source_file key (e.g. "2025-09-10") to match speaker_map keys
        data["source_file"] = f.stem.replace("_diarized", "")
        transcripts.append(data)
    return transcripts


def load_speaker_map(speaker_map_path: str) -> dict:
    with open(speaker_map_path) as f:
        return json.load(f)


def build_candidate_corpora(transcripts: List[Dict], speaker_map: Dict) -> Dict[str, List[str]]:
    # speaker_map keys are "source_file/SPEAKER_XX" — build a lookup per source_file
    # so we can match against the bare SPEAKER_XX labels in transcript segments
    file_speaker_to_name = {}  # {source_file: {speaker_id: name}}
    for key, info in speaker_map.items():
        if info.get("include_in_quiz") and info.get("role") == "candidate" and info.get("name"):
            source_file = info.get("source_file", "")
            speaker_id = info.get("speaker_id", "")
            file_speaker_to_name.setdefault(source_file, {})[speaker_id] = info["name"]

    all_names = {name for mapping in file_speaker_to_name.values() for name in mapping.values()}
    corpora = {name: [] for name in all_names}

    for transcript in transcripts:
        # Derive source_file key from the transcript's file field or first segment metadata
        source_file = transcript.get("source_file") or transcript.get("date", "")
        speaker_to_name = file_speaker_to_name.get(source_file, {})

        for seg in transcript.get("segments", []):
            speaker = seg.get("speaker", "")
            text = seg.get("text", "").strip()
            if speaker in speaker_to_name and text:
                corpora[speaker_to_name[speaker]].append(text)

    return corpora


def main():
    transcript_dir = "data/transcripts"
    speaker_map_path = "data/speaker_map.json"
    output_path = "data/candidate_corpora.json"

    print("[1/3] Loading transcripts...")
    transcripts = load_transcripts(transcript_dir)
    if not transcripts:
        print(f"ERROR: No transcripts found in {transcript_dir}. Run transcribe.py first.")
        return
    print(f"      Loaded {len(transcripts)} transcript(s)")

    print("[2/3] Loading speaker map...")
    if not Path(speaker_map_path).exists():
        print(f"ERROR: {speaker_map_path} not found. Run identify_speakers.py first.")
        return
    speaker_map = load_speaker_map(speaker_map_path)

    candidates = [
        info["name"]
        for info in speaker_map.values()
        if info.get("include_in_quiz") and info.get("role") == "candidate" and info.get("name")
    ]
    if len(candidates) < 2:
        print(f"ERROR: Need at least 2 candidates with include_in_quiz=true in {speaker_map_path}")
        print(f"       Found: {candidates}")
        return
    print(f"      Candidates: {candidates}")

    print("[3/3] Building candidate corpora...")
    corpora = build_candidate_corpora(transcripts, speaker_map)

    stats = {}
    for name, utterances in corpora.items():
        word_count = sum(len(u.split()) for u in utterances)
        stats[name] = {"utterance_count": len(utterances), "word_count": word_count}

    output = {"candidates": corpora, "stats": stats}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone! Corpora saved to: {output_path}")
    print("\nPer-candidate stats:")
    for name, s in stats.items():
        print(f"  {name}: {s['utterance_count']} utterances, {s['word_count']:,} words")
    print(f"\nSanity-check {output_path}, then run: python src/generate_quiz.py")


if __name__ == "__main__":
    main()
