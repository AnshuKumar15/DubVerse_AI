import os
import json
import argparse

from modules.audio_utils import extract_audio
from modules.transcribe import transcribe_audio, save_transcript
from modules.translate import translate_text


# ==============================
# Utility: Merge Whisper Segments
# ==============================
def merge_segments(segments, max_gap=1.0):
    """
    Merge nearby whisper segments to improve translation quality.
    max_gap: maximum silence gap (seconds) allowed between segments to merge.
    """
    if not segments:
        return []

    merged = []
    buffer = segments[0]

    for seg in segments[1:]:
        if seg["start"] - buffer["end"] <= max_gap:
            buffer["text"] += " " + seg["text"]
            buffer["end"] = seg["end"]
        else:
            merged.append(buffer)
            buffer = seg

    merged.append(buffer)
    return merged


# ==============================
# Translation Pipeline
# ==============================
def translate_transcript(input_json, output_json):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["segments"]

    print("Merging small segments for better context...")
    segments = merge_segments(segments)

    translated_segments = []

    print("Translating segments...")
    for seg in segments:
        english_text = seg["text"].strip()

        hindi_text = translate_text(english_text)

        translated_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "english": english_text,
            "hindi": hindi_text
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"segments": translated_segments}, f, indent=4, ensure_ascii=False)

    print(f"Hindi transcript saved at: {output_json}")


# ==============================
# Main Pipeline
# ==============================
def main(args):
    os.makedirs("data/temp", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)

    audio_path = "data/temp/audio.wav"
    transcript_path = "data/output/transcript.json"
    hindi_transcript_path = "data/output/hindi_transcript.json"

    # Step 1: Extract Audio
    print("\n=== Step 1: Extracting Audio ===")
    extract_audio(args.video_path, audio_path)

    # Step 2: Transcription
    print("\n=== Step 2: Transcribing Audio ===")
    result = transcribe_audio(audio_path, model_size=args.whisper_model)
    save_transcript(result, transcript_path)

    # Step 3: Translation
    print("\n=== Step 3: Translating to Hindi ===")
    translate_transcript(transcript_path, hindi_transcript_path)

    print("\n✅ Day 2 Pipeline Complete.")


# ==============================
# CLI Entry
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supernan Dubbing Pipeline - Day 1 & 2")

    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video file"
    )

    parser.add_argument(
        "--whisper_model",
        type=str,
        default="base",
        help="Whisper model size: tiny, base, small, medium, large"
    )

    args = parser.parse_args()

    main(args)