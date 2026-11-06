"""
Centralized Resource and Writable Path Abstraction for MediaProof.

Handles path resolution in both:
1. Normal development mode (relative to project root).
2. PyInstaller frozen mode (read-only resources from sys._MEIPASS, persistent data in sys.executable parent).
"""

import sys
from pathlib import Path

# Base Project Root in Development Mode
DEV_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    """Returns True if running inside a PyInstaller frozen application."""
    return getattr(sys, "frozen", False)


def get_bundle_dir() -> Path:
    """Returns the directory containing read-only bundled application resources.
    In frozen mode, this resolves to sys._MEIPASS (or sys.executable parent).
    In dev mode, this resolves to DEV_PROJECT_ROOT.
    """
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        return Path(sys.executable).resolve().parent
    return DEV_PROJECT_ROOT


def get_portable_root() -> Path:
    """Returns the directory where persistent writable application data resides.
    In frozen mode, this resolves to the directory containing the executable.
    In dev mode, this resolves to DEV_PROJECT_ROOT.
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return DEV_PROJECT_ROOT


# ------------------------------------------------------------------------------
# Read-Only Resource Paths (Bundled inside app)
# ------------------------------------------------------------------------------

def get_frontend_dir() -> Path:
    """Returns path to static frontend directory."""
    return get_bundle_dir() / "frontend"


def get_tools_ffmpeg_dir() -> Path:
    """Returns path to portable FFmpeg binaries directory."""
    return get_bundle_dir() / "tools" / "ffmpeg"


def get_models_dir() -> Path:
    """Returns path to base models directory."""
    return get_bundle_dir() / "models"


def get_video_models_dir() -> Path:
    """Returns path to video watermark models directory."""
    return get_models_dir() / "video"


def get_forensics_models_dir() -> Path:
    """Returns path to forensic ML models directory."""
    return get_models_dir() / "forensics"


def get_face_detector_models_dir() -> Path:
    """Returns path to YuNet face detector model directory."""
    return get_forensics_models_dir() / "face_detector"


# ------------------------------------------------------------------------------
# Persistent Writable Paths (Stored outside _MEIPASS, persistent across runs)
# ------------------------------------------------------------------------------

def get_db_dir() -> Path:
    """Returns path to persistent database directory."""
    p = get_portable_root() / "database"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_db_path() -> Path:
    """Returns path to persistent SQLite database file.
    Respects MEDIAPROOF_DB_PATH environment variable override when configured (e.g. during test isolation).
    """
    import os
    env_override = os.environ.get("MEDIAPROOF_DB_PATH")
    if env_override:
        return Path(env_override)
    return get_db_dir() / "mediaproof.db"


def get_protected_data_dir() -> Path:
    """Returns path to protected media output directory."""
    p = get_portable_root() / "data" / "protected"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_temp_data_dir() -> Path:
    """Returns path to temporary file directory."""
    p = get_portable_root() / "data" / "temp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logs_dir() -> Path:
    """Returns path to application logs directory."""
    p = get_portable_root() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p
