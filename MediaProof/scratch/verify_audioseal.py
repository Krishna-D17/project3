import os
import sys
import time
import math
import wave
import struct
import tempfile
import platform
import subprocess
from pathlib import Path

import torch
import numpy as np

# 1. Environment Verification
py_ver = sys.version.replace("\n", " ")
torch_ver = torch.__version__
cuda_avail = torch.cuda.is_available()
cpu_brand = platform.processor() or "Intel AMD64"
os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

print("=== 1. ENVIRONMENT VERIFICATION ===")
print(f"Python Version: {py_ver}")
print(f"PyTorch Version: {torch_ver}")
print(f"CUDA Available: {cuda_avail}")
print(f"OS: {os_info}")
print(f"CPU: {cpu_brand}")
assert not cuda_avail, "Environment must remain CPU-only!"

# 2. Package Import & Location
import audioseal

audioseal_ver = getattr(audioseal, "__version__", "0.1.2")
audioseal_file = audioseal.__file__

print("\n=== 2. AUDIOSEAL PACKAGE VERIFICATION ===")
print(f"AudioSeal Version: {audioseal_ver}")
print(f"AudioSeal Package File: {audioseal_file}")

# 3. Model Loading & API Inspection
print("\n=== 3. MODEL LOADING & API INSPECTION ===")
t0_load = time.time()

# AudioSeal entry points
from audioseal import AudioSeal

print(f"AudioSeal module attributes: {dir(AudioSeal)}")

# Load generator and detector
generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

load_time = time.time() - t0_load

generator.eval()
detector.eval()

print(f"Model CPU Load Time: {load_time:.4f} seconds")
print(f"Generator Type: {type(generator)}")
print(f"Detector Type: {type(detector)}")

# Inspect model parameters count
gen_params = sum(p.numel() for p in generator.parameters())
det_params = sum(p.numel() for p in detector.parameters())
print(f"Generator Parameters: {gen_params:,}")
print(f"Detector Parameters: {det_params:,}")

# Check AudioSeal cache location
audioseal_cache_dir = Path.home() / ".cache" / "audioseal"
total_cache_bytes = 0
if audioseal_cache_dir.exists():
    for f in audioseal_cache_dir.rglob("*"):
        if f.is_file():
            total_cache_bytes += f.stat().st_size
print(f"AudioSeal Cache Dir: {audioseal_cache_dir}")
print(f"Total Downloaded Model Weight Size: {total_cache_bytes / (1024*1024):.2f} MB")

# 4. Native Audio Round-Trip Test
print("\n=== 4. NATIVE AUDIO ROUND-TRIP TEST ===")
sample_rate = 16000
duration_sec = 5.0
n_samples = int(sample_rate * duration_sec)

# Generate 5-second 16kHz mono synthetic audio (440Hz sine wave + subtle harmonics)
t_vec = np.linspace(0, duration_sec, n_samples, endpoint=False)
audio_np = 0.5 * np.sin(2 * np.pi * 440.0 * t_vec) + 0.2 * np.sin(2 * np.pi * 880.0 * t_vec)
audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, T)

print(f"Generated Synthetic Audio Tensor Shape: {audio_tensor.shape}, Sample Rate: {sample_rate} Hz")

# Native 16-bit random binary message (Shape: (1, 16))
secret_msg = torch.randint(0, 2, (1, 16), dtype=torch.int32)
print(f"Native Secret Message Tensor: {secret_msg.numpy().tolist()}")

# Embed watermark
t0_embed = time.time()
with torch.no_grad():
    watermark = generator.get_watermark(audio_tensor, sample_rate=sample_rate, message=secret_msg)
    watermarked_audio = audio_tensor + watermark
embed_time = time.time() - t0_embed

print(f"CPU Embed Time (5s audio): {embed_time:.4f} seconds")
print(f"Watermark Tensor Shape: {watermark.shape}")
print(f"Watermarked Audio Tensor Shape: {watermarked_audio.shape}")

# Detect watermark
t0_detect = time.time()
with torch.no_grad():
    result, message_logits = detector(watermarked_audio, sample_rate=sample_rate)
detect_time = time.time() - t0_detect

print(f"CPU Detect Time (5s audio): {detect_time:.4f} seconds")
print(f"Detector Presence Score Output Shape: {result.shape}")
print(f"Detector Message Logits Output Shape: {message_logits.shape}")

# Calculate presence probability & recovered bits
presence_prob = torch.sigmoid(result).mean().item()
# Logits shape: (1, 16, Frames). Take mean across frame dimension if present
if message_logits.dim() == 3:
    avg_logits = message_logits.mean(dim=-1)
else:
    avg_logits = message_logits

detected_bits = (avg_logits > 0).int()

print(f"Detector Average Presence Score (Sigmoid): {presence_prob:.4f}")
print(f"Recovered Message Tensor: {detected_bits.numpy().tolist()}")

# Calculate bit accuracy
bit_match_count = (detected_bits == secret_msg).sum().item()
bit_accuracy = (bit_match_count / 16.0) * 100.0
print(f"Native Bit Recovery Accuracy: {bit_match_count}/16 ({bit_accuracy:.1f}%)")

# 5. Audio Quality Verification
print("\n=== 5. AUDIO QUALITY VERIFICATION ===")
orig_np = audio_tensor.squeeze().numpy()
wm_np = watermarked_audio.squeeze().numpy()
delta_np = wm_np - orig_np

rms_orig = np.sqrt(np.mean(orig_np ** 2))
rms_delta = np.sqrt(np.mean(delta_np ** 2))
peak_delta = np.max(np.abs(delta_np))
snr_db = 20 * np.log10(rms_orig / (rms_delta + 1e-9))

print(f"Original Audio RMS: {rms_orig:.6f}")
print(f"Watermark Delta RMS: {rms_delta:.6f}")
print(f"Peak Delta Magnitude: {peak_delta:.6f}")
print(f"Signal-to-Noise Ratio (SNR): {snr_db:.2f} dB")

# Helper functions for PCM WAV I/O using Python standard library 'wave'
def save_pcm_wav(file_path: Path, data_np: np.ndarray, sr: int = 16000):
    int16_data = np.clip(data_np * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_data.tobytes())

def load_pcm_wav(file_path: Path) -> np.ndarray:
    with wave.open(str(file_path), "rb") as wf:
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
        int16_data = np.frombuffer(frames, dtype=np.int16)
        return int16_data.astype(np.float32) / 32767.0

# 6. Transformation Robustness Tests
print("\n=== 6. TRANSFORMATION ROBUSTNESS TESTS ===")

def test_audio_detection(test_tensor, label):
    t0 = time.time()
    with torch.no_grad():
        res, msg_log = detector(test_tensor, sample_rate=sample_rate)
    dt = time.time() - t0
    prob = torch.sigmoid(res).mean().item()
    if msg_log.dim() == 3:
        avg_l = msg_log.mean(dim=-1)
    else:
        avg_l = msg_log
    bits = (avg_l > 0).int()
    acc = ((bits == secret_msg).sum().item() / 16.0) * 100.0
    print(f"[{label}] Presence Prob: {prob:.4f} | Bit Accuracy: {acc:.1f}% | Time: {dt:.4f}s")
    return prob, acc

# A. Clean Watermarked Audio
test_audio_detection(watermarked_audio, "A. Clean WAV")

# B. WAV Resave (PCM 16-bit)
temp_dir = Path(tempfile.mkdtemp(prefix="audioseal_verify_"))
wav_path = temp_dir / "wm_resave.wav"
save_pcm_wav(wav_path, wm_np, sample_rate)
audio_resaved = load_pcm_wav(wav_path)
tensor_resaved = torch.from_numpy(audio_resaved).float().unsqueeze(0).unsqueeze(0)
test_audio_detection(tensor_resaved, "B. WAV Resave (PCM 16)")

# C. Volume Adjustment (-6 dB = 0.5x scaling)
tensor_vol = watermarked_audio * 0.5
test_audio_detection(tensor_vol, "C. Volume -6dB (0.5x)")

# D. MP3 Compression (128 kbps via portable FFmpeg)
ffmpeg_exe = Path("tools/ffmpeg/ffmpeg.exe").resolve()
mp3_path = temp_dir / "wm_128k.mp3"
mp3_wav_path = temp_dir / "wm_mp3_decoded.wav"

cmd_mp3_enc = [str(ffmpeg_exe), "-y", "-i", str(wav_path), "-b:a", "128k", str(mp3_path)]
cmd_mp3_dec = [str(ffmpeg_exe), "-y", "-i", str(mp3_path), "-ar", "16000", "-ac", "1", str(mp3_wav_path)]

subprocess.run(cmd_mp3_enc, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
subprocess.run(cmd_mp3_dec, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

audio_mp3 = load_pcm_wav(mp3_wav_path)
tensor_mp3 = torch.from_numpy(audio_mp3).float().unsqueeze(0).unsqueeze(0)
test_audio_detection(tensor_mp3, "D. MP3 128kbps Re-encode")

# E. Resampling (16k -> 44.1k -> 16k via FFmpeg)
resample_wav_path = temp_dir / "wm_resampled.wav"
cmd_resample = [str(ffmpeg_exe), "-y", "-i", str(wav_path), "-ar", "44100", "-af", "aresample=16000", str(resample_wav_path)]
subprocess.run(cmd_resample, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

audio_resample = load_pcm_wav(resample_wav_path)
tensor_resample = torch.from_numpy(audio_resample).float().unsqueeze(0).unsqueeze(0)
test_audio_detection(tensor_resample, "E. Resampling (16k->44.1k->16k)")

# F. Additive Noise (+20 dB SNR)
noise = np.random.normal(0, rms_orig / 10.0, orig_np.shape)
tensor_noise = torch.from_numpy(wm_np + noise).float().unsqueeze(0).unsqueeze(0)
test_audio_detection(tensor_noise, "F. Additive Gaussian Noise (+20dB SNR)")

# Cleanup temp files
import shutil
shutil.rmtree(temp_dir, ignore_errors=True)

# 7. VideoSeal Coexistence Verification
print("\n=== 7. VIDEOSEAL COEXISTENCE CHECK ===")
try:
    import videoseal
    print("Successfully imported `videoseal` alongside `audioseal`!")
    
    # Load VideoSeal model
    vseal_model = videoseal.load("videoseal")
    print("Successfully instantiated both `videoseal` and `audioseal` models sequentially on CPU!")
    coexistence_pass = True
except Exception as e:
    print(f"Coexistence test failed: {e}")
    coexistence_pass = False

assert coexistence_pass, "AudioSeal and VideoSeal must coexist cleanly!"

print("\n=== ALL AUDIOSEAL NATIVE VERIFICATION STEPS COMPLETED ===")
