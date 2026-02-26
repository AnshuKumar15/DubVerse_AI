"""
Lip-sync module using Wav2Lip and optional face restoration (GFPGAN).
Generates a video where the speaker's lips match the dubbed Hindi audio.
"""

import os
import subprocess
import sys
import shutil


# ==============================
# Wav2Lip Integration
# ==============================

WAV2LIP_REPO = "https://github.com/Rudrabha/Wav2Lip.git"
WAV2LIP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Wav2Lip")


def ensure_wav2lip_installed():
    """
    Clone Wav2Lip repo if not already present.
    Returns the path to the Wav2Lip directory.
    """
    if not os.path.exists(WAV2LIP_DIR):
        print(f"  Cloning Wav2Lip repository...")
        subprocess.run(
            ["git", "clone", WAV2LIP_REPO, WAV2LIP_DIR],
            check=True
        )
        print(f"  Wav2Lip cloned → {WAV2LIP_DIR}")
    else:
        print(f"  Wav2Lip found at {WAV2LIP_DIR}")

    return WAV2LIP_DIR


def run_wav2lip(
    video_path,
    audio_path,
    output_path,
    checkpoint_path,
    resize_factor=1,
    pad=None,
    fps=None,
):
    """
    Run Wav2Lip inference to generate a lip-synced video.

    Args:
        video_path: Path to the input video (face video).
        audio_path: Path to the dubbed Hindi audio.
        output_path: Where to save the lip-synced video.
        checkpoint_path: Path to the Wav2Lip pretrained checkpoint (.pth).
        resize_factor: Downscale factor for processing (1 = original).
        pad: Face detection padding [top, bottom, left, right].
        fps: Override FPS (None = detect from source).

    Returns:
        Path to the lip-synced output video.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Wav2Lip checkpoint not found: {checkpoint_path}\n"
            f"Download from: https://github.com/Rudrabha/Wav2Lip#getting-the-weights\n"
            f"Place wav2lip_gan.pth in the 'checkpoints/' directory."
        )

    wav2lip_dir = ensure_wav2lip_installed()
    inference_script = os.path.join(wav2lip_dir, "inference.py")

    if not os.path.exists(inference_script):
        raise FileNotFoundError(f"Wav2Lip inference.py not found at {inference_script}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if pad is None:
        pad = [0, 10, 0, 0]

    # Build the inference command
    command = [
        sys.executable, inference_script,
        "--checkpoint_path", checkpoint_path,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--resize_factor", str(resize_factor),
        "--pads", str(pad[0]), str(pad[1]), str(pad[2]), str(pad[3]),
        "--nosmooth",  # Avoid temporal smoothing artifacts on short clips
    ]

    if fps is not None:
        command.extend(["--fps", str(fps)])

    print(f"  Running Wav2Lip inference...")
    print(f"    Video: {video_path}")
    print(f"    Audio: {audio_path}")
    print(f"    Output: {output_path}")

    result = subprocess.run(
        command,
        cwd=wav2lip_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"  Wav2Lip STDERR:\n{result.stderr}")
        raise RuntimeError(f"Wav2Lip inference failed with exit code {result.returncode}")

    if not os.path.exists(output_path):
        # Wav2Lip may output to results/ in its own directory
        default_output = os.path.join(wav2lip_dir, "results", "result_voice.mp4")
        if os.path.exists(default_output):
            shutil.move(default_output, output_path)
        else:
            raise FileNotFoundError(
                f"Wav2Lip did not produce output at {output_path}"
            )

    print(f"  Lip-sync video → {output_path}")
    return output_path


# ==============================
# Face Restoration (GFPGAN)
# ==============================

def restore_faces_gfpgan(input_video, output_video, gfpgan_model_path=None):
    """
    Apply GFPGAN face restoration to enhance face quality after Wav2Lip.
    Wav2Lip often produces blurry faces — GFPGAN sharpens them.

    This processes the video frame-by-frame:
    1. Extract frames from video
    2. Run GFPGAN on each frame
    3. Re-encode frames into video

    Args:
        input_video: Path to the Wav2Lip output video.
        output_video: Path to save the restored video.
        gfpgan_model_path: Path to GFPGAN model weights.

    Returns:
        Path to the restored video.
    """
    try:
        import cv2
        from gfpgan import GFPGANer
    except ImportError:
        print("  GFPGAN not installed. Skipping face restoration.")
        print("  Install with: pip install gfpgan")
        shutil.copy(input_video, output_video)
        return output_video

    if gfpgan_model_path and not os.path.exists(gfpgan_model_path):
        print(f"  GFPGAN model not found at {gfpgan_model_path}")
        print("  Skipping face restoration.")
        shutil.copy(input_video, output_video)
        return output_video

    print("  Running GFPGAN face restoration...")

    # Initialize GFPGAN
    restorer = GFPGANer(
        model_path=gfpgan_model_path or "GFPGANv1.4.pth",
        upscale=1,
        arch="clean",
        channel_multiplier=2,
    )

    # Process video frame by frame
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    temp_video = output_video + ".temp.mp4"
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # GFPGAN enhance
        _, _, restored_frame = restorer.enhance(
            frame,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )

        writer.write(restored_frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"    Restored {frame_idx}/{total_frames} frames...")

    cap.release()
    writer.release()

    # Re-mux with original audio
    command = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", input_video,
        "-c:v", "libx264",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output_video
    ]
    subprocess.run(command, check=True, capture_output=True)

    # Cleanup temp
    if os.path.exists(temp_video):
        os.remove(temp_video)

    print(f"  Restored video → {output_video}")
    return output_video


# ==============================
# Convenience: Full Lip-Sync Pipeline
# ==============================

def lipsync_video(
    video_path,
    audio_path,
    output_path,
    checkpoint_path,
    resize_factor=1,
    pad=None,
    fps=None,
    face_restore=False,
    gfpgan_model_path=None,
):
    """
    Full lip-sync pipeline: Wav2Lip + optional GFPGAN.

    Args:
        video_path: Input video with the face.
        audio_path: Dubbed Hindi audio.
        output_path: Final output video path.
        checkpoint_path: Wav2Lip checkpoint path.
        resize_factor: Downscale factor.
        pad: Face detection padding.
        fps: Override FPS.
        face_restore: Whether to run GFPGAN face restoration.
        gfpgan_model_path: Path to GFPGAN weights.

    Returns:
        Path to the final output video.
    """
    if face_restore:
        # First generate lip-synced video, then restore faces
        lipsync_raw = output_path.replace(".mp4", "_raw_lipsync.mp4")
        run_wav2lip(
            video_path=video_path,
            audio_path=audio_path,
            output_path=lipsync_raw,
            checkpoint_path=checkpoint_path,
            resize_factor=resize_factor,
            pad=pad,
            fps=fps,
        )
        restore_faces_gfpgan(lipsync_raw, output_path, gfpgan_model_path)
    else:
        run_wav2lip(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            resize_factor=resize_factor,
            pad=pad,
            fps=fps,
        )

    return output_path
