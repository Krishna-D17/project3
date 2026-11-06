import ctypes
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from media.ffmpeg import get_ffmpeg_status


def get_ram_gb() -> str:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            gb = stat.ullTotalPhys / (1024 ** 3)
            return f"{gb:.2f} GB"
    except Exception:
        pass
    return "Unknown"


def get_cpu_info() -> str:
    processor = platform.processor()
    if not processor:
        processor = os.environ.get("PROCESSOR_IDENTIFIER", "Generic CPU")
    return processor


def is_venv() -> str:
    in_venv = sys.prefix != sys.base_prefix or hasattr(sys, "real_prefix")
    return "YES" if in_venv else "NO"


def check_cuda_packages() -> str:
    cuda_pkgs = ["torch", "torch.cuda", "cupy", "pycuda"]
    detected = []
    for pkg in cuda_pkgs:
        if pkg in sys.modules:
            detected.append(pkg)
    if detected:
        return f"YES ({', '.join(detected)})"
    return "NO"


def main():
    python_ver = sys.version.split()[0]
    python_exe = sys.executable
    venv_status = is_venv()
    os_name = f"{platform.system()} {platform.release()} (Build {platform.version()})"
    arch = platform.machine()
    cpu_name = get_cpu_info()
    cpu_cores = os.cpu_count() or 1
    ram_str = get_ram_gb()

    cuda_pkg = check_cuda_packages()
    cuda_avail = "NO"
    gpu_detected = "NO"

    ffmpeg_status = get_ffmpeg_status()

    valid = venv_status == "YES" and sys.version_info >= (3, 9)
    env_status = "VALID (CPU-only)" if valid else "INVALID (Not inside .venv)"

    print("MediaProof Environment Check")
    print("============================")
    print("")
    print(f"Python version: {python_ver}")
    print(f"Python executable: {python_exe}")
    print(f"Virtual environment: {venv_status}")
    print(f"Operating system: {os_name}")
    print(f"Architecture: {arch}")
    print(f"CPU: {cpu_name}")
    print(f"CPU cores: {cpu_cores}")
    print(f"RAM: {ram_str}")
    print("")
    print(f"CUDA package detected: {cuda_pkg}")
    print(f"CUDA available: {cuda_avail}")
    print(f"GPU detected: {gpu_detected}")
    print("")
    print(f"FFmpeg available: {'YES' if ffmpeg_status['ffmpeg_available'] else 'NO'}")
    print(f"FFprobe available: {'YES' if ffmpeg_status['ffprobe_available'] else 'NO'}")
    print(f"FFmpeg path: {ffmpeg_status['ffmpeg_path'] or 'None'}")
    print(f"FFprobe path: {ffmpeg_status['ffprobe_path'] or 'None'}")
    print(f"FFmpeg version: {ffmpeg_status['version']}")
    print("")
    print(f"Environment status: {env_status}")

    if not valid:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
