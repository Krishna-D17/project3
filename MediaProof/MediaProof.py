import os
import sys
import time
import socket
import webbrowser
import threading
from pathlib import Path

# Path Resolution Strategy for Portable Application
if getattr(sys, 'frozen', False):
    # PyInstaller executable directory
    PORTABLE_ROOT = Path(sys.executable).resolve().parent
    BUNDLE_DIR = Path(getattr(sys, '_MEIPASS', PORTABLE_ROOT)).resolve()
else:
    PORTABLE_ROOT = Path(__file__).resolve().parent
    BUNDLE_DIR = PORTABLE_ROOT

# Ensure sys.path includes app and root directories
for path_dir in [BUNDLE_DIR, PORTABLE_ROOT, BUNDLE_DIR / "app", PORTABLE_ROOT / "app"]:
    if path_dir.exists() and str(path_dir) not in sys.path:
        sys.path.insert(0, str(path_dir))

# Enforce environment variables for local CPU-only offline operation
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure runtime directories exist
(PORTABLE_ROOT / "data" / "temp").mkdir(parents=True, exist_ok=True)
(PORTABLE_ROOT / "data" / "protected").mkdir(parents=True, exist_ok=True)
(PORTABLE_ROOT / "database").mkdir(parents=True, exist_ok=True)
(PORTABLE_ROOT / "logs").mkdir(parents=True, exist_ok=True)


def find_free_port(start_port: int = 8000, max_tries: int = 50) -> int:
    """Selects an available local port starting at start_port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


def open_browser_when_ready(url: str):
    """Opens default browser after server initializes."""
    time.sleep(1.5)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    print("==================================================")
    print("  MediaProof — Local Media Provenance & Verification")
    print("==================================================")
    print(f"Portable Root: {PORTABLE_ROOT}")

    port = find_free_port(8000)
    url = f"http://127.0.0.1:{port}/"
    print(f"Server URL: {url}")
    print("Starting local CPU backend server...")

    threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()

    import uvicorn
    from backend.main import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
