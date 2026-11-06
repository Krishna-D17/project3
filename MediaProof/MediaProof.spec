# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

block_cipher = None

PROJECT_ROOT = Path(SPECPATH).resolve()
if PROJECT_ROOT.name == 'packaging':
    PROJECT_ROOT = PROJECT_ROOT.parent.resolve()

datas = [
    (str(PROJECT_ROOT / 'frontend'), 'frontend'),
    (str(PROJECT_ROOT / 'tools' / 'ffmpeg'), 'tools/ffmpeg'),
    (str(PROJECT_ROOT / 'models'), 'models'),
]

binaries = []

hiddenimports = [
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'fastapi',
    'starlette',
    'pydantic',
    'torch',
    'torchvision',
    'torchvision.ops',
    'torchaudio',
    'videoseal',
    'audioseal',
    'timm',
    'omegaconf',
    'scipy',
    'scipy.fft',
    'scipy.signal',
    'cv2',
    'PIL',
    'sqlite3',
    'wave',
    'struct',
]

# Explicitly collect all data, binaries, and hidden imports for key packages
for pkg in ['torch', 'torchvision', 'torchaudio', 'videoseal', 'audioseal', 'timm', 'omegaconf', 'cv2', 'decord', 'einops', 'onnxruntime', 'julius']:
    d, b, h = collect_all(pkg)
    datas.extend(d)
    binaries.extend(b)
    hiddenimports.extend(h)

# Guarantee explicit collection of all native extension binaries (.pyd, .dll) in torchvision
try:
    import torchvision
    tv_dir = Path(torchvision.__file__).parent
    for f in tv_dir.rglob('*'):
        if f.is_file() and f.suffix.lower() in ['.pyd', '.dll', '.so']:
            rel_dir = str(f.relative_to(tv_dir.parent).parent)
            binaries.append((str(f), rel_dir))
except Exception as e:
    print(f"Warning collecting torchvision binaries: {e}")

# Guarantee explicit collection of decord files into _internal/decord/
try:
    import decord
    dec_dir = Path(decord.__file__).parent
    for f in dec_dir.rglob('*'):
        if f.is_file():
            rel_path = str(Path('decord') / f.relative_to(dec_dir))
            datas.append((str(f), str(Path(rel_path).parent)))
except Exception as e:
    print(f"Warning collecting decord files: {e}")

a = Analysis(
    [str(PROJECT_ROOT / 'MediaProof.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'notebook'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MediaProof',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='MediaProof',
)
