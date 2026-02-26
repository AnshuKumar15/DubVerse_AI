# DubVerse AI — Hindi Video Dubbing Pipeline

An end-to-end Python pipeline that takes an English video and produces a **Hindi-dubbed version** with voice cloning and lip synchronization — built entirely with open-source models at **₹0 cost**.

> **Supernan AI Intern Challenge** — "The Golden 15 Seconds"

---

## Pipeline Architecture

```
Input Video (.mp4)
    │
    ├── 1. SEGMENT EXTRACTION (FFmpeg)
    │       Extract 15-30s clip + audio + speaker reference
    │
    ├── 2. TRANSCRIPTION (OpenAI Whisper)
    │       Speech → English text with word-level timestamps
    │
    ├── 3. TRANSLATION (IndicTrans2 1B)
    │       English → Hindi, context-aware, batch-processed
    │
    ├── 4. VOICE CLONING + TTS (Coqui XTTS v2)
    │       Hindi text → Hindi speech matching original speaker voice
    │       Duration-adjusted to match original segment timing
    │
    ├── 5. LIP SYNC (Wav2Lip GAN)
    │       Lip movements re-generated to match Hindi audio
    │
    └── 6. FACE RESTORATION (GFPGAN) [Optional]
            Sharpen face quality degraded by Wav2Lip
    
Output: Hindi-dubbed video (.mp4)
```

---

## Quick Start

### Prerequisites

- **Python 3.10–3.12** (recommended: 3.11)
- **FFmpeg** installed and on PATH ([download](https://ffmpeg.org/download.html))
- **Git** for cloning Wav2Lip
- **CUDA GPU** recommended (works on CPU but very slow)

### Installation

```bash
# Clone the repository
git clone https://github.com/AnshuKumar15/DubVerse_AI.git
cd DubVerse_AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Model Checkpoints

Checkpoints are **downloaded automatically** when you first run the pipeline. Or download them manually:

```bash
# Download all required models (~850MB total)
python download_models.py

# Or selectively:
python download_models.py --wav2lip   # Wav2Lip GAN + S3FD face detector
python download_models.py --gfpgan    # GFPGAN face restoration
```

Models are saved to `checkpoints/`. If automatic download fails, get them manually:
- **wav2lip_gan.pth**: [Wav2Lip releases](https://github.com/Rudrabha/Wav2Lip#getting-the-weights)
- **GFPGANv1.4.pth**: [GFPGAN releases](https://github.com/TencentARC/GFPGAN/releases)
- **s3fd.pth**: Face detector (downloaded automatically with Wav2Lip)
```

### Run the Pipeline

```bash
# Process the default 15-30 second segment
python dub_video.py --video_path data/input/supernan_training.mp4

# Custom segment with better transcription
python dub_video.py --video_path data/input/supernan_training.mp4 \
    --start 15 --end 30 \
    --whisper_model small

# Skip lip-sync (audio-only dub)
python dub_video.py --video_path data/input/supernan_training.mp4 --no_lipsync

# Skip face restoration
python dub_video.py --video_path data/input/supernan_training.mp4 --no_face_restore
```

### Output

The dubbed video is saved to:
```
data/output/<video_name>_dubbed_hindi.mp4
```

Intermediate files (transcript, translated text, TTS segments) are in `data/temp/` and `data/output/`.

---

## Project Structure

```
DubVerse_AI/
├── dub_video.py              # Main pipeline orchestrator (CLI entry point)
├── config.py                 # Central configuration (paths, models, params)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── modules/
│   ├── audio_utils.py        # FFmpeg wrappers (extract, segment, merge, concat)
│   ├── transcribe.py         # Whisper speech-to-text with word timestamps
│   ├── translate.py          # IndicTrans2 English→Hindi translation
│   ├── tts.py                # XTTS v2 voice cloning + duration adjustment
│   └── lipsync.py            # Wav2Lip inference + GFPGAN face restoration
│
├── checkpoints/              # Model weights (not committed to git)
│   ├── wav2lip_gan.pth
│   └── GFPGANv1.4.pth
│
└── data/
    ├── input/                # Source videos
    ├── temp/                 # Intermediate processing files
    └── output/               # Final dubbed videos + transcripts
```

---

## Dependencies

| Package | Purpose | Cost |
|---------|---------|------|
| [OpenAI Whisper](https://github.com/openai/whisper) | Speech-to-text transcription | Free (open-source) |
| [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) | English → Hindi translation | Free (open-source) |
| [Coqui XTTS v2](https://github.com/coqui-ai/TTS) | Voice cloning + Hindi TTS | Free (open-source) |
| [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | Lip synchronization | Free (open-source) |
| [GFPGAN](https://github.com/TencentARC/GFPGAN) | Face restoration | Free (open-source) |
| [FFmpeg](https://ffmpeg.org/) | Audio/video manipulation | Free (open-source) |
| [librosa](https://librosa.org/) | Audio time-stretching | Free (open-source) |

**Total cost: ₹0**

---

## Estimated Cost Per Minute at Scale

| Component | Time (1 min video, A100 GPU) | Cost (cloud GPU) |
|-----------|------|------|
| Whisper (large) | ~10s | ~₹0.50 |
| IndicTrans2 (1B) | ~15s | ~₹0.75 |
| XTTS v2 voice cloning | ~60s | ~₹3.00 |
| Wav2Lip inference | ~45s | ~₹2.25 |
| GFPGAN face restoration | ~30s | ~₹1.50 |
| FFmpeg (encode/decode) | ~5s | ~₹0.25 |
| **Total per minute** | **~2.5 min** | **~₹8.25** |

*Based on AWS g5.xlarge (A10G) at ~₹150/hr. Colab Pro (A100): ~₹75/hr.*

---

## Scaling to 500 Hours Overnight

### Architecture for Production Scale

```
                    ┌──────────────┐
                    │  Job Queue   │
                    │  (Redis/SQS) │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │  Worker 1  │   │  Worker 2  │   │ Worker N   │
    │  (A100)    │   │  (A100)    │   │  (A100)    │
    └───────────┘   └───────────┘   └───────────┘
```

**Key modifications for 500 hours overnight:**

1. **Batch Processing**: Split videos into 30-second chunks with overlap. Process chunks in parallel across multiple GPU workers.

2. **Model Serving**: Deploy models as persistent services (e.g., Triton Inference Server) instead of loading per-video. Eliminates ~2min model load overhead per video.

3. **Pipeline Parallelism**: While TTS runs for video N, Whisper transcribes video N+1. Each pipeline stage runs independently.

4. **Spot Instances**: Use AWS Spot/GCP Preemptible (~70% cheaper). Checkpoint progress to resume on interruption.

5. **Storage**: Stream from/to S3. Avoid local disk bottlenecks.

6. **Estimated resources for 500hr overnight (12hr window)**:
   - 500hr × 2.5 min/min = ~1,250 GPU-hours
   - With 100 parallel A100 workers: ~12.5 hours
   - Cost: ~₹1,50,000 (at spot pricing ~₹120/hr/GPU)

---

## Known Limitations

1. **IndicTrans2 translation** can be literal for complex sentences. A fine-tuned model or post-editing step would improve naturalness.

2. **Wav2Lip** produces slightly blurry mouth regions. GFPGAN helps but isn't perfect — CodeFormer may yield better results.

3. **Voice cloning quality** depends heavily on the speaker reference clip. A clean, 6-10 second clip with minimal background noise gives best results.

4. **Audio duration matching** uses time-stretching which can distort speech at extreme ratios (>2x or <0.5x). For production, re-generating with adjusted speaking rate is better.

5. **Single speaker only** — does not handle multi-speaker scenarios or speaker diarization.

---

## What I'd Improve With More Time

- [ ] **Speaker diarization** (pyannote.audio) for multi-speaker videos
- [ ] **CodeFormer** instead of GFPGAN for sharper face restoration
- [ ] **VideoReTalking** as an alternative to Wav2Lip for higher fidelity lip sync
- [ ] **Punctuation restoration** before translation for better context
- [ ] **SSML support** in TTS for controlling emphasis, pauses, and speaking rate
- [ ] **A/B testing framework** to compare translation quality (IndicTrans2 vs NLLB vs Google)
- [ ] **WebUI** with Gradio for non-technical users
- [ ] **Docker container** for reproducible deployment
- [ ] **Streaming pipeline** for real-time dubbing

---

## License

This project uses open-source models and tools. See individual model licenses:
- Whisper: MIT License
- IndicTrans2: MIT License  
- Coqui TTS: MPL-2.0
- Wav2Lip: Custom academic license
- GFPGAN: Apache-2.0