from TTS.api import TTS
import soundfile as sf
import librosa
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def generate_hindi_speech(text, speaker_wav, output_path):
    """
    text: Hindi text
    speaker_wav: short clip of original speaker voice
    output_path: where generated wav will be saved
    """

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="hi",
        file_path=output_path
    )

    return output_path

def adjust_audio_duration(audio_path, target_duration, output_path):
    y, sr = librosa.load(audio_path, sr=None)

    current_duration = librosa.get_duration(y=y, sr=sr)

    speed_factor = current_duration / target_duration

    y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)

    sf.write(output_path, y_stretched, sr)

    return output_path