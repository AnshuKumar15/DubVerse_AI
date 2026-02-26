"""
Speech-to-text transcription using OpenAI Whisper.
Supports word-level timestamps for precise audio alignment.
"""

import whisper
import json
import os
import torch
import gc


def transcribe_audio(audio_path, model_size="base", word_timestamps=True, language=None):
    """
    Transcribe audio using Whisper.

    Args:
        audio_path: Path to the WAV audio file.
        model_size: Whisper model size (tiny/base/small/medium/large).
        word_timestamps: If True, include word-level timing info.
        language: Force source language (e.g. 'en', 'kn'). Auto-detected if None.

    Returns:
        Whisper result dict with segments (and word timestamps if enabled).
    """
    print(f"  Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    transcribe_opts = {
        "word_timestamps": word_timestamps,
        "verbose": False,
    }
    if language:
        transcribe_opts["language"] = language
        print(f"  Forced language: {language}")

    print(f"  Transcribing: {audio_path}")
    result = model.transcribe(audio_path, **transcribe_opts)

    print(f"  Detected language: {result.get('language', 'unknown')}")
    print(f"  Total segments: {len(result.get('segments', []))}")

    # Free GPU memory — essential for small GPUs (4GB)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("  Whisper model unloaded (GPU memory freed)")

    return result


def save_transcript(result, output_path):
    """Save Whisper transcription result to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"  Transcript saved → {output_path}")


def load_transcript(input_path):
    """Load a previously saved transcript JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_segments_in_range(result, start_time, end_time):
    """
    Filter transcript segments to only those within a time range.
    Adjusts segment timestamps to be relative to the segment start.

    Args:
        result: Whisper result dict or loaded transcript.
        start_time: Start of range in seconds (in original video time).
        end_time: End of range in seconds (in original video time).

    Returns:
        List of segments with timestamps adjusted to be relative to start_time.
    """
    segments = result.get("segments", [])
    filtered = []

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Keep segments that overlap with our range
        if seg_end > start_time and seg_start < end_time:
            adjusted = {
                "start": max(0, seg_start - start_time),
                "end": min(end_time - start_time, seg_end - start_time),
                "text": seg["text"].strip(),
            }

            # Preserve word-level timestamps if available
            if "words" in seg:
                adjusted["words"] = []
                for word in seg["words"]:
                    w_start = word.get("start", 0)
                    w_end = word.get("end", 0)
                    if w_end > start_time and w_start < end_time:
                        adjusted["words"].append({
                            "word": word["word"],
                            "start": max(0, w_start - start_time),
                            "end": min(end_time - start_time, w_end - start_time),
                        })

            filtered.append(adjusted)

    return filtered