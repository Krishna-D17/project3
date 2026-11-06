import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.paths import get_tools_ffmpeg_dir


class FFmpegError(Exception):
    """Base exception for FFmpeg operations."""
    pass


class FFmpegNotFoundError(FFmpegError):
    """Raised when ffmpeg or ffprobe executable is not found."""
    pass


class FFmpegExecutionError(FFmpegError):
    """Raised when an ffmpeg/ffprobe subprocess execution fails."""
    def __init__(self, message: str, returncode: int, stdout: str, stderr: str):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FFmpegTimeoutError(FFmpegError):
    """Raised when an ffmpeg/ffprobe subprocess times out."""
    pass


def get_ffmpeg_path() -> Optional[Path]:
    """Resolves the path to the ffmpeg executable."""
    tools_dir = get_tools_ffmpeg_dir()
    # 1. Check portable directory
    local_ffmpeg = tools_dir / "ffmpeg.exe"
    if local_ffmpeg.is_file():
        return local_ffmpeg

    # Linux/Mac fallback if executable without extension exists in tools dir
    local_ffmpeg_unix = tools_dir / "ffmpeg"
    if local_ffmpeg_unix.is_file():
        return local_ffmpeg_unix

    # 2. Check System PATH
    system_path = shutil.which("ffmpeg")
    if system_path:
        return Path(system_path)

    return None


def get_ffprobe_path() -> Optional[Path]:
    """Resolves the path to the ffprobe executable."""
    tools_dir = get_tools_ffmpeg_dir()
    # 1. Check portable directory
    local_ffprobe = tools_dir / "ffprobe.exe"
    if local_ffprobe.is_file():
        return local_ffprobe

    local_ffprobe_unix = tools_dir / "ffprobe"
    if local_ffprobe_unix.is_file():
        return local_ffprobe_unix

    # 2. Check System PATH
    system_path = shutil.which("ffprobe")
    if system_path:
        return Path(system_path)

    return None


def run_ffmpeg_command(cmd_args: List[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Executes a subprocess command safely without shell=True."""
    if not cmd_args:
        raise ValueError("cmd_args must be a non-empty list of strings")

    try:
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False
        )
        return result
    except subprocess.TimeoutExpired as e:
        raise FFmpegTimeoutError(f"Subprocess timed out after {timeout} seconds: {' '.join(cmd_args)}") from e
    except Exception as e:
        raise FFmpegError(f"Failed to execute process ({' '.join(cmd_args)}): {e}") from e


def get_ffmpeg_version(binary_path: Path) -> str:
    """Gets version string from an ffmpeg or ffprobe executable."""
    try:
        res = run_ffmpeg_command([str(binary_path), "-version"], timeout=5.0)
        if res.returncode == 0 and res.stdout:
            first_line = res.stdout.splitlines()[0]
            return first_line
    except Exception:
        pass
    return "Unknown"


def get_ffmpeg_status() -> Dict[str, object]:
    """Returns availability and resolved paths for ffmpeg and ffprobe."""
    ffmpeg_p = get_ffmpeg_path()
    ffprobe_p = get_ffprobe_path()

    ffmpeg_avail = ffmpeg_p is not None and ffmpeg_p.exists()
    ffprobe_avail = ffprobe_p is not None and ffprobe_p.exists()

    version_str = "Unavailable"
    if ffprobe_avail:
        version_str = get_ffmpeg_version(ffprobe_p)
    elif ffmpeg_avail:
        version_str = get_ffmpeg_version(ffmpeg_p)

    return {
        "ffmpeg_available": ffmpeg_avail,
        "ffprobe_available": ffprobe_avail,
        "ffmpeg_path": str(ffmpeg_p) if ffmpeg_p else None,
        "ffprobe_path": str(ffprobe_p) if ffprobe_p else None,
        "version": version_str,
    }
