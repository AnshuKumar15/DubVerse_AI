"""
Supernan AI Dubbing Pipeline — dub_video.py

End-to-end pipeline to dub an English video into Hindi:
  1. Extract & segment the video
  2. Transcribe audio (Whisper)
  3. Translate English → Hindi (IndicTrans2)
  4. Generate Hindi speech with voice cloning (XTTS v2)
  5. Lip-sync the video to dubbed audio (Wav2Lip)
  6. (Optional) Face restoration (GFPGAN)

Usage:
  python dub_video.py --video_path input.mp4
  python dub_video.py --video_path input.mp4 --start 15 --end 30 --whisper_model small
"""

import os
import json
import argparse

from config import (
    SEGMENT_START, SEGMENT_END, WHISPER_MODEL_SIZE, WHISPER_WORD_TIMESTAMPS,
    WAV2LIP_CHECKPOINT, WAV2LIP_RESIZE_FACTOR, WAV2LIP_PAD, WAV2LIP_FPS,
    FACE_RESTORE_ENABLED, GFPGAN_MODEL_PATH, SPEAKER_REF_DURATION,
    get_pipeline_paths, ensure_dirs,
)

from modules.audio_utils import (
    extract_audio, extract_video_segment, extract_audio_segment,
    merge_audio_video, get_media_duration,
)
from modules.transcribe import (
    transcribe_audio, save_transcript, get_segments_in_range,
)
from modules.translate import (
    translate_text, translate_segments, save_translated_transcript,
    unload_model as unload_translate_model,
)
from modules.tts import (
    generate_segment_audio, stitch_segments_with_timing,
)
from modules.lipsync import lipsync_video


# ==============================
# Utility: Merge Whisper Segments
# ==============================
def merge_segments(segments, max_gap=1.0):
    """
    Merge nearby whisper segments to improve translation quality.
    Prevents sending tiny fragments to the translator.

    Args:
        segments: List of segment dicts with 'start', 'end', 'text'.
        max_gap: Maximum silence gap (seconds) allowed between segments to merge.

    Returns:
        List of merged segments.
    """
    if not segments:
        return []

    merged = []
    buffer = dict(segments[0])  # shallow copy

    for seg in segments[1:]:
        if seg["start"] - buffer["end"] <= max_gap:
            buffer["text"] += " " + seg["text"]
            buffer["end"] = seg["end"]
        else:
            merged.append(buffer)
            buffer = dict(seg)

    merged.append(buffer)
    return merged


# ==============================
# Main Pipeline
# ==============================
def main(args):
    ensure_dirs()

    # Auto-download model checkpoints if missing
    if not os.path.exists(WAV2LIP_CHECKPOINT):
        print("\nModel checkpoints not found. Downloading automatically...")
        from download_models import download_all
        download_all()

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    paths = get_pipeline_paths(video_name)

    start = args.start
    end = args.end
    segment_duration = end - start

    print("=" * 60)
    print(f"  Supernan AI Dubbing Pipeline")
    print(f"  Input:   {args.video_path}")
    print(f"  Segment: {start}s – {end}s ({segment_duration}s)")
    print(f"  Whisper: {args.whisper_model}")
    print("=" * 60)

    # --------------------------------------------------
    # Step 1: Extract Video Segment
    # --------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 1: Extracting Video Segment")
    print("=" * 50)

    segment_video = paths["segment_video"]
    extract_video_segment(args.video_path, start, end, segment_video)

    # Extract audio from the segment
    segment_audio = paths["segment_audio"]
    extract_audio(segment_video, segment_audio)

    # Extract a speaker reference clip for voice cloning
    speaker_ref = paths["speaker_ref"]
    ref_end = min(SPEAKER_REF_DURATION, segment_duration)
    extract_audio_segment(segment_audio, 0, ref_end, speaker_ref)

    # --------------------------------------------------
    # Step 2: Transcription
    # --------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 2: Transcribing Audio (Whisper)")
    print("=" * 50)

    result = transcribe_audio(
        segment_audio,
        model_size=args.whisper_model,
        word_timestamps=WHISPER_WORD_TIMESTAMPS,
        language=args.language,
    )
    save_transcript(result, paths["transcript"])

    # Detect source language from Whisper
    detected_lang = result.get("language", args.language or "en")
    print(f"  Source language: {detected_lang}")

    segments = result.get("segments", [])
    print(f"  Found {len(segments)} raw segments")

    # --------------------------------------------------
    # Step 3: Merge & Translate
    # --------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 3: Translating to Hindi (NLLB-200)")
    print("=" * 50)

    # Merge small segments for better translation context
    merged = merge_segments(segments, max_gap=1.0)
    print(f"  Merged into {len(merged)} segments for translation")

    translated = translate_segments(merged, source_lang=detected_lang)
    save_translated_transcript(translated, paths["hindi_transcript"])

    # Free translation model before loading TTS
    unload_translate_model()

    # --------------------------------------------------
    # Step 4: Hindi Voice Generation (TTS + Voice Cloning)
    # --------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 4: Generating Hindi Speech (XTTS v2)")
    print("=" * 50)

    tts_dir = paths["tts_segments_dir"]
    os.makedirs(tts_dir, exist_ok=True)

    segment_results = generate_segment_audio(
        segments=translated,
        speaker_wav=speaker_ref,
        output_dir=tts_dir,
    )

    # Stitch all TTS segments into a single audio track with correct timing
    hindi_audio = paths["hindi_audio"]
    stitch_segments_with_timing(segment_results, segment_duration, hindi_audio)

    # --------------------------------------------------
    # Step 5: Lip Sync (Wav2Lip + GFPGAN)
    # --------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 5: Lip-Syncing Video (Wav2Lip)")
    print("=" * 50)

    if os.path.exists(WAV2LIP_CHECKPOINT):
        lipsync_video(
            video_path=segment_video,
            audio_path=hindi_audio,
            output_path=paths["final_output"],
            checkpoint_path=WAV2LIP_CHECKPOINT,
            resize_factor=WAV2LIP_RESIZE_FACTOR,
            pad=WAV2LIP_PAD,
            fps=WAV2LIP_FPS,
            face_restore=FACE_RESTORE_ENABLED,
            gfpgan_model_path=GFPGAN_MODEL_PATH,
        )
    else:
        print(f"  WARNING: Wav2Lip checkpoint not found at {WAV2LIP_CHECKPOINT}")
        print(f"  Falling back to audio-only merge (no lip sync).")
        merge_audio_video(segment_video, hindi_audio, paths["final_output"])

    # --------------------------------------------------
    # Done
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Output: {paths['final_output']}")
    print("=" * 60)

    return paths["final_output"]


# ==============================
# CLI Entry
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supernan AI Dubbing Pipeline — English to Hindi video dubbing"
    )

    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=SEGMENT_START,
        help=f"Start time in seconds (default: {SEGMENT_START})",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=SEGMENT_END,
        help=f"End time in seconds (default: {SEGMENT_END})",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default=WHISPER_MODEL_SIZE,
        choices=["tiny", "base", "small", "medium", "large"],
        help=f"Whisper model size (default: {WHISPER_MODEL_SIZE})",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Source language code (e.g. 'en', 'kn', 'hi'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--no_lipsync",
        action="store_true",
        help="Skip lip-sync, only merge audio with video",
    )
    parser.add_argument(
        "--no_face_restore",
        action="store_true",
        help="Skip GFPGAN face restoration",
    )

    args = parser.parse_args()
    main(args)