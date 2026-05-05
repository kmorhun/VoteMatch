# VoteMatch

A voter-matching quiz for the Cambridge School Committee 2025 election.
Voters answer ~10 Likert-scale questions about local education policy
and get matched to the candidates whose stated positions are closest to theirs.

**Key design principle:** Every inferred candidate position is grounded in a
verbatim quote from a public CCTV forum transcript. No position is published
without a supporting quote.

---

## Architecture

```
VoteMatch/
├── data/
│   ├── 2025-09-10-CCTV.mp3        ← raw audio (not committed to git)
│   ├── transcripts/
│   │   └── 2025-09-10_diarized.json   ← output of Step 1
│   ├── speaker_map.json               ← output of Step 2 (edit this!)
│   └── quiz_data.json                 ← output of Step 3
│
├── src/
│   ├── transcribe.py          ← Step 1: WhisperX + pyannote diarization
│   ├── identify_speakers.py   ← Step 2: heuristic speaker → name mapping
│   └── generate_quiz.py       ← Step 3: Claude API → grounded quiz questions
│
├── docs/                      ← GitHub Pages site (static, no server needed)
│   ├── index.html             ← the quiz UI
│   └── quiz_data.json         ← copy of data/quiz_data.json
│
├── models/                    ← Whisper model weights (auto-downloaded, not committed)
├── requirements.txt
└── .env                       ← HF_TOKEN and ANTHROPIC_API_KEY (never commit this!)
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Install `ffmpeg` as described on https://github.com/openai/whisper#setup

```bash
# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

> **Note on `pyannote.audio`:** You must accept the model license on Hugging Face:
>
> - Go to https://huggingface.co/pyannote/speaker-diarization-community-1 and input the correct information to Agree to the model license
> - Go to https://huggingface.co/pyannote/segmentation-3.0 and click "Agree"

### 2. Set up environment variables

Create a `.env` file in the project root:

```
HF_TOKEN=hf_your_token_here
ANTHROPIC_API_KEY=sk-ant-your_key_here
```

Get your Hugging Face token at https://huggingface.co/settings/tokens
Get your Anthropic API key at https://console.anthropic.com

---

## Running the Pipeline

### Step 1: Transcribe and diarize

```bash
python src/transcribe.py --audio data/2025-09-10-CCTV.mp3
```

**Output:** `data/transcripts/2025-09-10_diarized.json`

> ⏱ **CPU timing:** Expect ~30–60 minutes per 4–6 hour recording on CPU.
> The script uses Whisper `large-v3-turbo` with `int8` quantization for maximum
> CPU efficiency. This is a one-time cost.

Repeat for each audio file:

```bash
python src/transcribe.py --audio data/2025-09-27-CCTV.mp3
python src/transcribe.py --audio data/2025-09-28-CCTV.mp3
```

---

### Step 2: Identify speakers

```bash
python src/identify_speakers.py --transcripts data/transcripts/
```

**Output:** `data/speaker_map.json`

This script uses heuristics (moderator introductions, self-identification)
to guess which `SPEAKER_XX` label corresponds to which candidate.
**You must review and edit this file.** Open it and:

- Set each `"name"` to the candidate's actual full name (or `null` for unknowns)
- Set `"role"` to `"candidate"` or `"moderator"`
- Set `"include_in_quiz"` to `false` for moderators and anyone not running

Example `speaker_map.json` after editing:

```json
{
  "SPEAKER_00": {
    "name": "Alice Johnson",
    "role": "moderator",
    "include_in_quiz": false,
    ...
  },
  "SPEAKER_01": {
    "name": "Bob Chen",
    "role": "candidate",
    "include_in_quiz": true,
    ...
  }
}
```

The `word_count` field is useful: the moderator is usually the highest-word-count
non-candidate speaker.

---

### Step 3: Generate quiz questions

```bash
python src/generate_quiz.py
```

**Output:** `data/quiz_data.json`

This calls the Claude API (`claude-opus-4-5`) to:

1. Identify ~10 recurring policy themes from the transcripts
2. Draft a quiz question for each theme
3. Predict each candidate's Likert response WITH a verbatim supporting quote

**Only questions where ≥2 candidates have real quotes are included.**
Questions with insufficient transcript evidence are automatically dropped.

---

### Step 4: Deploy to GitHub Pages

```bash
cp data/quiz_data.json docs/quiz_data.json
git add docs/
git commit -m "Update quiz data"
git push
```

Then enable GitHub Pages in your repo settings → Pages → Source: `main` branch, `/docs` folder.

Your quiz will be live at: `https://YOUR_USERNAME.github.io/YOUR_REPO/`

---

## Testing Locally

The `docs/` folder contains a working sample `quiz_data.json` with placeholder candidates.
To test the frontend before running the pipeline:

```bash
cd docs
python -m http.server 8080
# Visit http://localhost:8080
```

> ⚠️ You must use a local server (not `file://`) because the quiz fetches
> `quiz_data.json` via `fetch()`.

---

## Caveats and Transparency

- Candidate positions are **inferred** from public statements at CCTV forums.
  They may not reflect a candidate's full or current position.
- Speaker diarization is imperfect — verify the `speaker_map.json` carefully.
- The quiz shows verbatim quotes alongside every position to let voters judge
  for themselves whether the inference is fair.
- This tool is nonpartisan and does not endorse any candidate.

---

## Adding More Forum Recordings

When you have the Sep 27 and Sep 28 recordings:

1. Run `transcribe.py` on each new file
2. Re-run `identify_speakers.py` (it reads all `*_diarized.json` files at once)
3. Re-check `speaker_map.json` — new speakers from new dates will appear
4. Re-run `generate_quiz.py` to incorporate all transcripts
5. Copy to `docs/quiz_data.json` and redeploy
