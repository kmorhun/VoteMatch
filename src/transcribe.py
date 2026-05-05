"""
Step 1: Transcribe and diarize audio files.

Usage:
    python src/transcribe.py --audio data/2025-09-10-CCTV.mp3

Outputs:
    data/transcripts/2025-09-10_diarized.json

Requirements:
    pip install whisperx pyannote.audio torch

You'll need a HuggingFace token with access to:
    - pyannote/speaker-diarization-community-1
    - pyannote/segmentation-3.0
Set it in .env as HF_TOKEN=hf_...
"""

import argparse
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not set in .env file. See README for instructions.")


def transcribe_and_diarize(audio_path: str, output_dir: str = "data/transcripts") -> str:
    """
    Runs WhisperX transcription + pyannote speaker diarization on an audio file.

    Returns the path to the output JSON file.
    """
    import whisperx

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output filename from audio filename date
    date_str = re.search(r"\d{4}-\d{2}-\d{2}", audio_path.name)
    if date_str:
        out_name = f"{date_str.group()}_diarized.json"
    else:
        out_name = f"{audio_path.stem}_diarized.json"
    output_path = output_dir / out_name

    print(f"[1/4] Loading audio: {audio_path}")
    # WhisperX works on CPU — just slower than GPU
    device = "cpu"
    compute_type = "int8"  # int8 is much faster on CPU than float16

    print(f"[2/4] Loading Whisper model (large-v3-turbo)...")
    print("      This may take a few minutes on first run (downloading ~800MB).")
    model = whisperx.load_model(
        "large-v3-turbo",
        device,
        compute_type=compute_type,
        download_root="models/",
    )

    audio = whisperx.load_audio(str(audio_path))

    print(f"[3/4] Transcribing... (this will take a while on CPU — ~30-60 min for a 4-6hr file)")
    result = model.transcribe(audio, language="en", batch_size=4)  # small batch size for CPU
    print(f"      Detected language: {result['language']}")
    print(f"      Segments so far: {len(result['segments'])}")

    # Align timestamps
    print("      Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device, return_char_alignments=False
    )

    del model_a  # free memory

    print(f"[4/4] Running speaker diarization...")
    print("      Using pyannote/speaker-diarization-community-1 (requires HF token with model access)")
    diarize_model = DiarizationPipeline(token=HF_TOKEN, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Save output
    output = {
        "audio_file": str(audio_path),
        "language": result.get("language", "en"),
        "segments": result["segments"],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone! Saved to: {output_path}")
    print(f"Segments: {len(result['segments'])}")

    # Print a speake_r summary
    speakers = set()
    for seg in result["segments"]:
        if "speaker" in seg:
            speakers.add(seg["speaker"])
    print(f"Speakers detected: {sorted(speakers)}")

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe and diarize a CCTV forum recording.")
    parser.add_argument("--audio", required=True, help="Path to the audio file (mp3/wav/etc)")
    parser.add_argument(
        "--output-dir",
        default="data/transcripts",
        help="Directory to save the diarized transcript JSON (default: data/transcripts)",
    )
    args = parser.parse_args()
    transcribe_and_diarize(args.audio, args.output_dir)
