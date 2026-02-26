"""
Automatic model checkpoint downloader for the Supernan AI Dubbing Pipeline.
Downloads Wav2Lip GAN and GFPGAN v1.4 weights.

Usage:
    python download_models.py           # Download all models
    python download_models.py --wav2lip # Download Wav2Lip only
    python download_models.py --gfpgan  # Download GFPGAN only
"""

import os
import sys
import argparse
import urllib.request
import hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# ==============================
# Model Download Registry
# ==============================
MODELS = {
    "wav2lip_gan": {
        "filename": "wav2lip_gan.pth",
        "url": "hf://camenduru/Wav2Lip/checkpoints/wav2lip_gan.pth",
        "backup_urls": [],
        "description": "Wav2Lip GAN checkpoint for lip synchronization (~416MB)",
        "use_hf_hub": True,
        "hf_repo": "camenduru/Wav2Lip",
        "hf_path": "checkpoints/wav2lip_gan.pth",
    },
    "wav2lip": {
        "filename": "wav2lip.pth",
        "url": "hf://camenduru/Wav2Lip/checkpoints/wav2lip.pth",
        "backup_urls": [],
        "description": "Wav2Lip base checkpoint (alternative, lighter) (~416MB)",
        "use_hf_hub": True,
        "hf_repo": "camenduru/Wav2Lip",
        "hf_path": "checkpoints/wav2lip.pth",
    },
    "gfpgan": {
        "filename": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "backup_urls": [],
        "description": "GFPGAN v1.4 for face restoration (~350MB)",
    },
    "s3fd": {
        "filename": "s3fd.pth",
        "url": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
        "backup_urls": [
            "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2Lip/s3fd.pth",
        ],
        "description": "S3FD face detection model (required by Wav2Lip) (~90MB)",
    },
}

# Which models to download by default
DEFAULT_MODELS = ["wav2lip_gan", "gfpgan", "s3fd"]


def download_file(url, dest_path, description=""):
    """Download a file with progress indicator."""
    print(f"\n  Downloading: {description or os.path.basename(dest_path)}")
    print(f"  URL: {url}")
    print(f"  Saving to: {dest_path}")

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    try:
        # Custom progress hook
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(
                    f"\r  Progress: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)"
                )
            else:
                mb_down = downloaded / (1024 * 1024)
                sys.stdout.write(f"\r  Downloaded: {mb_down:.1f} MB")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print()  # newline after progress

        file_size = os.path.getsize(dest_path)
        print(f"  Done! File size: {file_size / (1024 * 1024):.1f} MB")
        return True

    except Exception as e:
        print(f"\n  Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def download_model(model_key):
    """Download a single model, trying backup URLs if primary fails."""
    if model_key not in MODELS:
        print(f"  Unknown model: {model_key}")
        return False

    model = MODELS[model_key]
    dest_path = os.path.join(CHECKPOINTS_DIR, model["filename"])

    # Skip if already downloaded
    if os.path.exists(dest_path):
        file_size = os.path.getsize(dest_path)
        print(f"  {model['filename']} already exists ({file_size / (1024 * 1024):.1f} MB) — skipping")
        return True

    # Use HuggingFace Hub if specified
    if model.get("use_hf_hub"):
        try:
            from huggingface_hub import hf_hub_download
            print(f"\n  Downloading via HuggingFace Hub: {model['description']}")
            print(f"    Repo: {model['hf_repo']}, File: {model['hf_path']}")
            downloaded = hf_hub_download(
                repo_id=model["hf_repo"],
                filename=model["hf_path"],
                local_dir=CHECKPOINTS_DIR + "/_hf_tmp",
            )
            # Move to correct location
            import shutil
            shutil.move(downloaded, dest_path)
            # Cleanup temp dir
            tmp_dir = os.path.join(CHECKPOINTS_DIR, "_hf_tmp")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

            file_size = os.path.getsize(dest_path)
            print(f"  Done! File size: {file_size / (1024 * 1024):.1f} MB")
            return True
        except Exception as e:
            print(f"  HuggingFace Hub download failed: {e}")

    # Try primary URL (direct HTTP download)
    if not model.get("use_hf_hub") and download_file(model["url"], dest_path, model["description"]):
        return True

    # Try backup URLs
    for backup_url in model.get("backup_urls", []):
        print(f"  Trying backup URL...")
        if download_file(backup_url, dest_path, model["description"]):
            return True

    print(f"  FAILED: Could not download {model['filename']}")
    return False


def setup_wav2lip_face_detection(checkpoints_dir):
    """
    Copy the s3fd face detection model to where Wav2Lip expects it.
    Wav2Lip looks for it in: Wav2Lip/face_detection/detection/sfd/s3fd.pth
    """
    s3fd_src = os.path.join(checkpoints_dir, "s3fd.pth")
    if not os.path.exists(s3fd_src):
        return

    wav2lip_dir = os.path.join(BASE_DIR, "Wav2Lip")
    s3fd_dest_dir = os.path.join(wav2lip_dir, "face_detection", "detection", "sfd")
    s3fd_dest = os.path.join(s3fd_dest_dir, "s3fd.pth")

    if os.path.exists(wav2lip_dir) and not os.path.exists(s3fd_dest):
        os.makedirs(s3fd_dest_dir, exist_ok=True)
        import shutil
        shutil.copy2(s3fd_src, s3fd_dest)
        print(f"  Copied s3fd.pth → {s3fd_dest}")


def download_all(model_keys=None):
    """Download all required model checkpoints."""
    if model_keys is None:
        model_keys = DEFAULT_MODELS

    print("=" * 55)
    print("  Supernan AI — Model Checkpoint Downloader")
    print("=" * 55)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    results = {}
    for key in model_keys:
        results[key] = download_model(key)

    # Setup face detection for Wav2Lip
    setup_wav2lip_face_detection(CHECKPOINTS_DIR)

    # Summary
    print("\n" + "=" * 55)
    print("  Download Summary:")
    for key, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"    {MODELS[key]['filename']}: {status}")
    print("=" * 55)

    return all(results.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model checkpoints")
    parser.add_argument("--wav2lip", action="store_true", help="Download Wav2Lip GAN only")
    parser.add_argument("--gfpgan", action="store_true", help="Download GFPGAN only")
    parser.add_argument("--s3fd", action="store_true", help="Download S3FD face detector only")
    parser.add_argument("--all", action="store_true", help="Download all models")

    args = parser.parse_args()

    if args.wav2lip:
        download_all(["wav2lip_gan", "s3fd"])
    elif args.gfpgan:
        download_all(["gfpgan"])
    elif args.s3fd:
        download_all(["s3fd"])
    else:
        # Default: download all required models
        download_all()
