"""
Central configuration for the Supernan AI Dubbing Pipeline.
All paths, model settings, and pipeline parameters in one place.
"""

import os

# ==============================
# Project Directory Structure
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMP_DIR = os.path.join(DATA_DIR, "temp")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# ==============================
# Video Segment Configuration
# ==============================
# Which segment of the source video to process (seconds)
SEGMENT_START = 15.0  # 0:15
SEGMENT_END = 30.0    # 0:30

# ==============================
# Whisper (Transcription)
# ==============================
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
# Use word-level timestamps for precise alignment
WHISPER_WORD_TIMESTAMPS = True

# ==============================
# Translation (NLLB-200)
# ==============================
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
TRANSLATION_MAX_LENGTH = 512

# ==============================
# TTS / Voice Cloning (Coqui XTTS v2)
# ==============================
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_LANGUAGE = "hi"
# Duration of speaker reference clip for voice cloning (seconds)
SPEAKER_REF_DURATION = 10.0

# ==============================
# Lip Sync (Wav2Lip)
# ==============================
# Path to the Wav2Lip pretrained checkpoint
# Download from: https://github.com/Rudrabha/Wav2Lip#getting-the-weights
WAV2LIP_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "wav2lip_gan.pth")
# Resize factor for faster processing (1 = original, 2 = half, etc.)
WAV2LIP_RESIZE_FACTOR = 1
# Padding around detected face [top, bottom, left, right]
WAV2LIP_PAD = [0, 10, 0, 0]
# Output video FPS (None = match source)
WAV2LIP_FPS = None

# ==============================
# Face Restoration (GFPGAN / CodeFormer)
# ==============================
FACE_RESTORE_ENABLED = True
# Options: "gfpgan", "codeformer"
FACE_RESTORE_MODEL = "gfpgan"
GFPGAN_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "GFPGANv1.4.pth")

# ==============================
# FFmpeg Settings
# ==============================
FFMPEG_LOGLEVEL = "error"  # quiet, error, warning, info

# ==============================
# Pipeline File Paths (auto-generated)
# ==============================
def get_pipeline_paths(video_name="video"):
    """Return a dict of all intermediate and output file paths."""
    return {
        # Input
        "segment_video": os.path.join(TEMP_DIR, f"{video_name}_segment.mp4"),
        # Audio
        "full_audio": os.path.join(TEMP_DIR, f"{video_name}_full_audio.wav"),
        "segment_audio": os.path.join(TEMP_DIR, f"{video_name}_segment_audio.wav"),
        "speaker_ref": os.path.join(TEMP_DIR, f"{video_name}_speaker_ref.wav"),
        # Transcription
        "transcript": os.path.join(OUTPUT_DIR, f"{video_name}_transcript.json"),
        # Translation
        "hindi_transcript": os.path.join(OUTPUT_DIR, f"{video_name}_hindi_transcript.json"),
        # TTS
        "tts_segments_dir": os.path.join(TEMP_DIR, "tts_segments"),
        "hindi_audio": os.path.join(TEMP_DIR, f"{video_name}_hindi_audio.wav"),
        # Lip Sync
        "lipsync_video": os.path.join(TEMP_DIR, f"{video_name}_lipsync.mp4"),
        # Face Restoration
        "restored_video": os.path.join(TEMP_DIR, f"{video_name}_restored.mp4"),
        # Final Output
        "final_output": os.path.join(OUTPUT_DIR, f"{video_name}_dubbed_hindi.mp4"),
    }


def ensure_dirs():
    """Create all required directories."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
