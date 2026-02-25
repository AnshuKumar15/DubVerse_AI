import subprocess
import os

def extract_audio(video_path, output_audio_path):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_audio_path
    ]
    subprocess.run(command, check=True)
    return output_audio_path

def extract_audio_segment(input_audio, start, end, output_path):
    command = [
        "ffmpeg",
        "-i", input_audio,
        "-ss", str(start),
        "-to", str(end),
        "-c", "copy",
        output_path
    ]
    subprocess.run(command, check=True)
    return output_path   