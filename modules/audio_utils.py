"""
Audio & video manipulation utilities using FFmpeg.
Handles extraction, segmentation, merging, and duration queries.
"""

import subprocess
import os
import json


def _run_ffmpeg(command, description=""):
    """Run an FFmpeg command with error handling."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error during {description}:")
        print(e.stderr)
        raise


def get_media_duration(file_path):
    """Get duration of a media file in seconds using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        file_path
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def get_video_fps(video_path):
    """Get the FPS of a video file using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    fps_str = info["streams"][0]["r_frame_rate"]
    num, den = map(int, fps_str.split("/"))
    return num / den


def extract_audio(video_path, output_audio_path):
    """Extract full audio track from video as 16kHz mono WAV."""
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_audio_path
    ]
    _run_ffmpeg(command, "extract audio")
    print(f"  Audio extracted → {output_audio_path}")
    return output_audio_path


def extract_video_segment(video_path, start, end, output_path):
    """
    Extract a segment of video (with audio) between start and end seconds.
    Uses keyframe-accurate seeking for precision.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start),
        "-to", str(end),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]
    _run_ffmpeg(command, "extract video segment")
    print(f"  Video segment [{start}s - {end}s] → {output_path}")
    return output_path


def extract_audio_segment(input_audio, start, end, output_path):
    """Extract a time segment from an audio file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    command = [
        "ffmpeg", "-y",
        "-i", input_audio,
        "-ss", str(start),
        "-to", str(end),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    _run_ffmpeg(command, "extract audio segment")
    return output_path


def merge_audio_video(video_path, audio_path, output_path):
    """
    Replace the audio track of a video with a new audio file.
    The video stream is copied without re-encoding.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    _run_ffmpeg(command, "merge audio + video")
    print(f"  Merged audio+video → {output_path}")
    return output_path


def concatenate_audio_files(audio_paths, output_path):
    """
    Concatenate multiple audio files sequentially into one.
    Useful for stitching per-segment TTS outputs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a temporary file list for FFmpeg concat demuxer
    list_path = output_path + ".list.txt"
    with open(list_path, "w") as f:
        for path in audio_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    command = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    _run_ffmpeg(command, "concatenate audio")

    # Cleanup temp list
    if os.path.exists(list_path):
        os.remove(list_path)

    return output_path


def generate_silence(duration, output_path, sample_rate=16000):
    """Generate a silent WAV file of the given duration."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    command = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", str(duration),
        "-acodec", "pcm_s16le",
        output_path
    ]
    _run_ffmpeg(command, "generate silence")
    return output_path