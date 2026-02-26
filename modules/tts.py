"""
Text-to-Speech with Voice Cloning using Coqui XTTS v2.
Generates Hindi speech that mimics the original speaker's voice.
Includes audio duration adjustment for lip-sync alignment.
"""

import os
import soundfile as sf
import librosa
import numpy as np
import torch

# Lazy-loaded TTS model
_tts = None
_device = None


def _load_tts():
    """Lazy-load the XTTS v2 model on first use."""
    global _tts, _device
    if _tts is not None:
        return

    from TTS.api import TTS

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading XTTS v2 model on {_device}...")
    _tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(_device)
    print("  XTTS v2 model loaded.")


def generate_hindi_speech(text, speaker_wav, output_path):
    """
    Generate Hindi speech with voice cloning.

    Args:
        text: Hindi text to synthesize.
        speaker_wav: Path to a short WAV clip of the original speaker
                     (used as voice reference for cloning).
        output_path: Where to save the generated WAV.

    Returns:
        Path to the generated audio file.
    """
    _load_tts()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    _tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="hi",
        file_path=output_path
    )

    return output_path


def generate_segment_audio(segments, speaker_wav, output_dir, segment_duration_map=None):
    """
    Generate TTS audio for each translated segment.

    Args:
        segments: List of dicts with 'start', 'end', 'hindi' keys.
        speaker_wav: Path to speaker reference audio.
        output_dir: Directory to save individual segment WAVs.
        segment_duration_map: Optional dict mapping segment index to target duration.

    Returns:
        List of dicts with segment info and generated audio paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, seg in enumerate(segments):
        hindi_text = seg["hindi"].strip()
        if not hindi_text:
            continue

        seg_path = os.path.join(output_dir, f"segment_{i:03d}.wav")
        target_duration = seg["end"] - seg["start"]

        print(f"  TTS segment {i + 1}/{len(segments)}: \"{hindi_text[:50]}...\"")

        # Generate raw TTS audio
        generate_hindi_speech(hindi_text, speaker_wav, seg_path)

        # Adjust duration to match original segment timing
        adjusted_path = os.path.join(output_dir, f"segment_{i:03d}_adjusted.wav")
        adjust_audio_duration(seg_path, target_duration, adjusted_path)

        results.append({
            "index": i,
            "start": seg["start"],
            "end": seg["end"],
            "hindi": hindi_text,
            "raw_audio": seg_path,
            "adjusted_audio": adjusted_path,
            "target_duration": target_duration,
        })

    return results


def adjust_audio_duration(audio_path, target_duration, output_path):
    """
    Time-stretch or compress audio to match a target duration.
    Uses librosa phase vocoder for high-quality stretching.

    Args:
        audio_path: Path to input WAV.
        target_duration: Desired duration in seconds.
        output_path: Where to save the adjusted WAV.

    Returns:
        Path to the adjusted audio.
    """
    y, sr = librosa.load(audio_path, sr=None)
    current_duration = librosa.get_duration(y=y, sr=sr)

    if current_duration == 0 or target_duration == 0:
        sf.write(output_path, y, sr)
        return output_path

    speed_factor = current_duration / target_duration

    # Clamp speed factor to avoid extreme distortion
    speed_factor = max(0.5, min(2.0, speed_factor))

    if abs(speed_factor - 1.0) < 0.05:
        # No significant adjustment needed
        sf.write(output_path, y, sr)
    else:
        y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)

        # Trim or pad to exact target duration
        target_samples = int(target_duration * sr)
        if len(y_stretched) > target_samples:
            y_stretched = y_stretched[:target_samples]
        elif len(y_stretched) < target_samples:
            y_stretched = np.pad(y_stretched, (0, target_samples - len(y_stretched)))

        sf.write(output_path, y_stretched, sr)

    return output_path


def stitch_segments_with_timing(segment_results, total_duration, output_path, sr=16000):
    """
    Stitch individual TTS segments into a single audio track,
    placing each segment at its correct timestamp with silence in gaps.

    Args:
        segment_results: List of dicts from generate_segment_audio().
        total_duration: Total duration of the output audio in seconds.
        output_path: Where to save the final stitched WAV.
        sr: Sample rate (default: 16000).

    Returns:
        Path to the stitched audio file.
    """
    total_samples = int(total_duration * sr)
    output_audio = np.zeros(total_samples, dtype=np.float32)

    for seg in segment_results:
        y, seg_sr = librosa.load(seg["adjusted_audio"], sr=sr)
        start_sample = int(seg["start"] * sr)
        end_sample = start_sample + len(y)

        # Ensure we don't exceed total length
        if end_sample > total_samples:
            y = y[:total_samples - start_sample]
            end_sample = total_samples

        output_audio[start_sample:end_sample] = y

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, output_audio, sr)
    print(f"  Stitched Hindi audio → {output_path}")

    return output_path