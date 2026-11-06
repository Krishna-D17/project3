import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from forensics.service import ForensicService

TEST_AUDIO = PROJECT_ROOT / "data" / "temp" / "test_media" / "test_audio.wav"
TEST_VIDEO = PROJECT_ROOT / "data" / "temp" / "test_media" / "test_video.mp4"
TEST_COMB = PROJECT_ROOT / "data" / "temp" / "test_media" / "test_combined.mp4"

def run_benchmarks():
    print("=== FORENSICS PERFORMANCE BENCHMARK (CPU-ONLY) ===")

    # 1. Benchmark 1s WAV Audio Signal & Metadata Extraction
    t0 = time.time()
    res_audio = ForensicService.analyze_media(TEST_AUDIO)
    t_audio = time.time() - t0
    print(f"1. Audio (WAV) Analysis: {t_audio:.4f}s | Signals: {len(res_audio.audio_signals)} | Risk: {res_audio.risk_score}")

    # 2. Benchmark 1s MP4 Video Signal & Metadata Extraction
    t0 = time.time()
    res_video = ForensicService.analyze_media(TEST_VIDEO)
    t_video = time.time() - t0
    print(f"2. Video (MP4) Analysis: {t_video:.4f}s | Signals: {len(res_video.video_signals)} | Risk: {res_video.risk_score}")

    # 3. Benchmark Combined MP4 Audio/Video Analysis
    t0 = time.time()
    res_comb = ForensicService.analyze_media(TEST_COMB)
    t_comb = time.time() - t0
    print(f"3. Combined (MP4) Analysis: {t_comb:.4f}s | Signals: {len(res_comb.video_signals) + len(res_comb.audio_signals) + len(res_comb.metadata_signals)} | Risk: {res_comb.risk_score}")

    print("=== BENCHMARK COMPLETE ===")

if __name__ == "__main__":
    run_benchmarks()
