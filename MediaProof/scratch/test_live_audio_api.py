import os
import sys
import time
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_AUDIO = PROJECT_ROOT / "data" / "temp" / "test_media" / "test_audio.wav"

def run_live_api_verification():
    print("=== STARTING LIVE AUDIO API VERIFICATION ===")

    # 1. Health check
    res = requests.get(f"{BASE_URL}/api/health")
    print(f"1. GET /api/health Status: {res.status_code}")
    print(res.json())
    print()

    # 2. Register provenance
    with open(TEST_AUDIO, "rb") as f:
        res = requests.post(
            f"{BASE_URL}/api/provenance/register",
            files={"file": ("live_audio.wav", f, "audio/wav")}
        )
    print(f"2. POST /api/provenance/register Status: {res.status_code}")
    reg_data = res.json()
    print(reg_data)
    asset_id = reg_data["asset_id"]
    print()

    # 3. Embed Custom Engine Watermark
    with open(TEST_AUDIO, "rb") as f:
        res = requests.post(
            f"{BASE_URL}/api/watermark/audio/embed",
            files={"file": ("live_audio.wav", f, "audio/wav")},
            data={"asset_id": asset_id, "engine": "custom"}
        )
    print(f"3. POST /api/watermark/audio/embed (custom) Status: {res.status_code}")
    embed_custom = res.json()
    print(embed_custom)
    wm_custom_path = Path(embed_custom["output_path"])
    print()

    # 4. Detect Custom Engine Watermark
    with open(wm_custom_path, "rb") as f:
        res = requests.post(
            f"{BASE_URL}/api/watermark/audio/detect",
            files={"file": ("wm_custom.wav", f, "audio/wav")},
            data={"engine": "custom"}
        )
    print(f"4. POST /api/watermark/audio/detect (custom) Status: {res.status_code}")
    print(res.json())
    print()

    # 5. Embed Meta AudioSeal Engine Watermark
    with open(TEST_AUDIO, "rb") as f:
        res = requests.post(
            f"{BASE_URL}/api/watermark/audio/embed",
            files={"file": ("live_audio.wav", f, "audio/wav")},
            data={"asset_id": asset_id, "engine": "audioseal"}
        )
    print(f"5. POST /api/watermark/audio/embed (audioseal) Status: {res.status_code}")
    embed_aseal = res.json()
    print(embed_aseal)
    wm_aseal_path = Path(embed_aseal["output_path"])
    print()

    # 6. Detect Meta AudioSeal Engine Watermark
    with open(wm_aseal_path, "rb") as f:
        res = requests.post(
            f"{BASE_URL}/api/watermark/audio/detect",
            files={"file": ("wm_audioseal.wav", f, "audio/wav")},
            data={"engine": "audioseal"}
        )
    print(f"6. POST /api/watermark/audio/detect (audioseal) Status: {res.status_code}")
    print(res.json())
    print()
    print("=== LIVE AUDIO API VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    run_live_api_verification()
