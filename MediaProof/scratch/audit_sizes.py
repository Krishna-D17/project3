import os
from pathlib import Path

def get_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.glob("**/*"):
        if p.is_file():
            total += p.stat().st_size
    return total

def fmt(num_bytes: int) -> str:
    mb = num_bytes / (1024 * 1024)
    if mb > 1024:
        return f"{mb / 1024:.2f} GB ({num_bytes:,} bytes)"
    return f"{mb:.2f} MB ({num_bytes:,} bytes)"

root = Path(r"c:\Users\krish\Desktop\project3\MediaProof")
dist_portable = root / "dist" / "MediaProof_Portable"
exe_path = dist_portable / "MediaProof.exe"
internal_path = dist_portable / "_internal"
torch_path = internal_path / "torch"
videoseal_ckpt = root / "ckpts" / "videoseal_y_256b_img.pth"
ffmpeg_path = root / "tools" / "ffmpeg"
frontend_path = root / "frontend"

exe_sz = get_size(exe_path)
internal_sz = get_size(internal_path)
torch_sz = get_size(torch_path)
videoseal_sz = get_size(videoseal_ckpt)
ffmpeg_sz = get_size(ffmpeg_path)
frontend_sz = get_size(frontend_path)
total_sz = get_size(dist_portable)

app_src_sz = sum(get_size(root / d) for d in ["backend", "watermark", "provenance", "forensics", "media"])

print("==================================================")
print("  MediaProof Step 7C Portable Build Size Audit")
print("==================================================")
print(f"EXE SIZE: {fmt(exe_sz)}")
print(f"PYTHON RUNTIME SIZE (_internal): {fmt(internal_sz)}")
print(f"PYTORCH SIZE: {fmt(torch_sz)}")
print(f"VIDEOSEAL MODEL SIZE: {fmt(videoseal_sz)}")
print(f"AUDIOSEAL MODEL SIZE: ~1.50 MB (bundled in package)")
print(f"FFMPEG SIZE: {fmt(ffmpeg_sz)}")
print(f"FRONTEND SIZE: {fmt(frontend_sz)}")
print(f"APPLICATION SOURCE SIZE: {fmt(app_src_sz)}")
print(f"TOTAL PORTABLE SIZE: {fmt(total_sz)}")
print("==================================================")
