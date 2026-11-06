import os
import sys
import shutil
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_FFMPEG_DIR = PROJECT_ROOT / "tools" / "ffmpeg"
TEMP_DIR = PROJECT_ROOT / "data" / "temp"

# Official static build source for Windows (GyanD codexffmpeg release asset)
FFMPEG_ZIP_URL = "https://github.com/GyanD/codexffmpeg/releases/download/7.0.2/ffmpeg-7.0.2-essentials_build.zip"


def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    sys.stdout.write(f"\rDownloading FFmpeg static build... {percent}%")
    sys.stdout.flush()


def download_and_extract_ffmpeg():
    TOOLS_FFMPEG_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = TOOLS_FFMPEG_DIR / "ffmpeg.exe"
    ffprobe_exe = TOOLS_FFMPEG_DIR / "ffprobe.exe"

    if ffmpeg_exe.exists() and ffprobe_exe.exists():
        print(f"FFmpeg and FFprobe already exist in {TOOLS_FFMPEG_DIR}")
        return True

    zip_path = TEMP_DIR / "ffmpeg_download.zip"
    print(f"Downloading FFmpeg from {FFMPEG_ZIP_URL}...")
    try:
        urllib.request.urlretrieve(FFMPEG_ZIP_URL, zip_path, reporthook=download_progress)
        print("\nDownload complete. Extracting binaries...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                filename = os.path.basename(file_info.filename)
                if filename.lower() in ("ffmpeg.exe", "ffprobe.exe"):
                    file_info.filename = filename
                    zip_ref.extract(file_info, TOOLS_FFMPEG_DIR)
                    print(f"Extracted {filename} -> {TOOLS_FFMPEG_DIR / filename}")

        if zip_path.exists():
            zip_path.unlink()

        if ffmpeg_exe.exists() and ffprobe_exe.exists():
            print("FFmpeg portable setup completed successfully!")
            return True
        else:
            print("[ERROR] Failed to extract ffmpeg.exe / ffprobe.exe")
            return False
    except Exception as e:
        print(f"\n[ERROR] Download/extraction failed: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False


if __name__ == "__main__":
    success = download_and_extract_ffmpeg()
    sys.exit(0 if success else 1)
