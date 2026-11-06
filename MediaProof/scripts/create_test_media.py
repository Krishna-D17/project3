import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from media.ffmpeg import get_ffmpeg_path, run_ffmpeg_command

TEST_MEDIA_DIR = PROJECT_ROOT / "data" / "temp" / "test_media"


def generate_test_media():
    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe or not ffmpeg_exe.exists():
        print("[SKIP] FFmpeg executable is not available. Cannot generate test media.")
        return False

    TEST_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    wav_path = TEST_MEDIA_DIR / "test_audio.wav"
    mp4_video_path = TEST_MEDIA_DIR / "test_video.mp4"
    mp4_combined_path = TEST_MEDIA_DIR / "test_combined.mp4"
    space_path = TEST_MEDIA_DIR / "test audio with spaces.wav"

    print(f"Generating test media files under {TEST_MEDIA_DIR}...")

    # 1. Generate short WAV audio (1s, 44100 Hz sine wave)
    cmd_wav = [
        str(ffmpeg_exe), "-y",
        "-f", "lavfi", "-i", "sine=f=440:r=44100",
        "-t", "1",
        str(wav_path)
    ]
    res_wav = run_ffmpeg_command(cmd_wav)
    if res_wav.returncode == 0 and wav_path.exists():
        print(f"Generated WAV audio: {wav_path.name} ({wav_path.stat().st_size} bytes)")
    else:
        print(f"[ERROR] Failed to generate WAV: {res_wav.stderr}")
        return False

    # 2. Generate short MP4 video (1s, 320x240 solid blue video)
    cmd_video = [
        str(ffmpeg_exe), "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=25",
        "-t", "1",
        "-pix_fmt", "yuv420p",
        str(mp4_video_path)
    ]
    res_video = run_ffmpeg_command(cmd_video)
    if res_video.returncode == 0 and mp4_video_path.exists():
        print(f"Generated MP4 video: {mp4_video_path.name} ({mp4_video_path.stat().st_size} bytes)")
    else:
        print(f"[ERROR] Failed to generate MP4 video: {res_video.stderr}")
        return False

    # 3. Generate short MP4 video + audio (1s video + audio track)
    cmd_combined = [
        str(ffmpeg_exe), "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:r=25",
        "-f", "lavfi", "-i", "sine=f=440:r=44100",
        "-t", "1",
        "-pix_fmt", "yuv420p",
        str(mp4_combined_path)
    ]
    res_combined = run_ffmpeg_command(cmd_combined)
    if res_combined.returncode == 0 and mp4_combined_path.exists():
        print(f"Generated MP4 combined: {mp4_combined_path.name} ({mp4_combined_path.stat().st_size} bytes)")
    else:
        print(f"[ERROR] Failed to generate MP4 combined: {res_combined.stderr}")
        return False

    # 4. Generate WAV file with space in path
    cmd_space = [
        str(ffmpeg_exe), "-y",
        "-f", "lavfi", "-i", "sine=f=880:r=44100",
        "-t", "1",
        str(space_path)
    ]
    res_space = run_ffmpeg_command(cmd_space)
    if res_space.returncode == 0 and space_path.exists():
        print(f"Generated space path audio: {space_path.name} ({space_path.stat().st_size} bytes)")

    print("Test media generation completed successfully!")
    return True


if __name__ == "__main__":
    success = generate_test_media()
    sys.exit(0 if success else 1)
