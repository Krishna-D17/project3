import os
import sys
import time
import subprocess
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if len(sys.argv) > 1:
    EXE_PATH = Path(sys.argv[1]).resolve()
else:
    EXE_PATH = BASE_DIR / "dist" / "MediaProof" / "MediaProof.exe"
BASE_URL = "http://127.0.0.1:8000"

def log(msg):
    print(f"[PACKAGING TEST] {msg}")

def test_packaged_app():
    if not EXE_PATH.exists():
        log(f"ERROR: Executable not found at {EXE_PATH}")
        sys.exit(1)

    log(f"Found packaged executable: {EXE_PATH} ({EXE_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # Test Media Paths
    test_video = BASE_DIR / "data" / "temp" / "test_media" / "test_video.mp4"
    test_audio = BASE_DIR / "data" / "temp" / "test_media" / "test_audio.wav"
    forensic_orig = BASE_DIR / "data" / "protected" / "wm_000_original_7s.mp4"
    forensic_deepfake = BASE_DIR / "data" / "protected" / "wm_000_virat_7s_from_protected.mp4"

    if not test_video.exists() or not test_audio.exists():
        log(f"ERROR: Test media files missing!")
        sys.exit(1)

    results = {}

    log("Starting packaged application subprocess...")
    proc = subprocess.Popen([str(EXE_PATH)], cwd=str(EXE_PATH.parent))

    try:
        # 1. Health check polling
        log("Waiting for backend health check...")
        healthy = False
        for _ in range(30):
            try:
                r = requests.get(f"{BASE_URL}/api/health", timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    healthy = True
                    break
            except Exception:
                pass
            time.sleep(1)

        if not healthy:
            log("ERROR: App failed to start within 30 seconds!")
            proc.kill()
            sys.exit(1)

        results["Health"] = "PASS"
        log("Backend is HEALTHY!")

        # 2. Status check
        r = requests.get(f"{BASE_URL}/api/media/status")
        status_data = r.json()
        log(f"Media Status response: {status_data}")

        ffmpeg_ok = status_data.get("ffmpeg_available", False) and status_data.get("ffprobe_available", False)
        results["FFmpeg"] = "PASS" if ffmpeg_ok else "FAIL"

        # 3. Provenance Register
        with open(test_video, "rb") as f:
            r = requests.post(f"{BASE_URL}/api/provenance/register", files={"file": ("test_video.mp4", f, "video/mp4")})
        assert r.status_code in (200, 201), f"Register failed: {r.text}"
        reg_data = r.json()
        asset_id = reg_data["asset_id"]
        results["SQLite / Provenance"] = "PASS"
        log(f"Registered Asset ID: {asset_id}")

        # 4. Custom Video Watermark
        with open(test_video, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/api/watermark/video/embed",
                data={"asset_id": asset_id, "engine": "custom"},
                files={"file": ("test_video.mp4", f, "video/mp4")}
            )
        assert r.status_code == 200, f"Custom embed failed: {r.text}"
        custom_embed_path = r.json()["output_path"]
        results["Custom Video WM"] = "PASS"

        # 5. VIDEOSEAL VIDEO WATERMARK EMBED (MANDATORY)
        log("Testing Meta VideoSeal embedding on 7s video...")
        vseal_input = forensic_orig if forensic_orig.exists() else test_video
        with open(vseal_input, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/api/watermark/video/embed",
                data={"asset_id": asset_id, "engine": "videoseal"},
                files={"file": (vseal_input.name, f, "video/mp4")}
            )
        if r.status_code != 200:
            log(f"VIDEOSEAL EMBED ERROR: {r.text}")
            results["VideoSeal Embed"] = "FAIL"
            results["VideoSeal Detect"] = "FAIL"
        else:
            vseal_res = r.json()
            assert vseal_res["engine"] == "videoseal", f"Unexpected engine: {vseal_res['engine']}"
            vseal_embed_file = vseal_res["output_path"]
            results["VideoSeal Embed"] = "PASS"
            log(f"VideoSeal Embedding succeeded! Engine: {vseal_res['engine']}")

            # 6. VIDEOSEAL VIDEO WATERMARK DETECT (MANDATORY)
            log("Testing Meta VideoSeal detection...")
            vseal_full_path = EXE_PATH.parent / vseal_embed_file
            if not vseal_full_path.exists():
                vseal_full_path = BASE_DIR / vseal_embed_file

            with open(vseal_full_path, "rb") as f:
                r = requests.post(
                    f"{BASE_URL}/api/watermark/video/detect",
                    data={"engine": "videoseal"},
                    files={"file": ("vseal_wm.mp4", f, "video/mp4")}
                )
            if r.status_code != 200:
                log(f"VIDEOSEAL DETECT ERROR: {r.text}")
                results["VideoSeal Detect"] = "FAIL"
            else:
                det_res = r.json()
                assert (det_res.get("watermark_detected") is True or det_res.get("detected") is True), f"Detection failed: {det_res}"
                assert det_res["engine"] == "videoseal"
                results["VideoSeal Detect"] = "PASS"
                log(f"VideoSeal Detection succeeded! Token: {det_res.get('payload_hex') or det_res.get('details', {}).get('watermark_token')}, Resolved Asset: {det_res.get('recovered_asset_id') or det_res.get('resolved_asset_id')}")

        # 7. Custom Audio Watermark
        with open(test_audio, "rb") as f:
            r = requests.post(f"{BASE_URL}/api/provenance/register", files={"file": ("test_audio.wav", f, "audio/wav")})
        assert r.status_code in (200, 201), f"Audio register failed: {r.text}"
        audio_asset_id = r.json()["asset_id"]

        with open(test_audio, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/api/watermark/audio/embed",
                data={"asset_id": audio_asset_id, "engine": "custom"},
                files={"file": ("test_audio.wav", f, "audio/wav")}
            )
        assert r.status_code == 200, f"Custom audio embed failed: {r.text}"
        results["Custom Audio WM"] = "PASS"

        # 8. AudioSeal Audio Watermark
        log("Testing Meta AudioSeal embedding & detection...")
        with open(test_audio, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/api/watermark/audio/embed",
                data={"asset_id": audio_asset_id, "engine": "audioseal"},
                files={"file": ("test_audio.wav", f, "audio/wav")}
            )
        assert r.status_code == 200, f"AudioSeal embed failed: {r.text}"
        aseal_file = r.json()["output_path"]
        aseal_full_path = EXE_PATH.parent / aseal_file if (EXE_PATH.parent / aseal_file).exists() else BASE_DIR / aseal_file

        with open(aseal_full_path, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/api/watermark/audio/detect",
                data={"engine": "audioseal"},
                files={"file": ("aseal_wm.wav", f, "audio/wav")}
            )
        assert r.status_code == 200 and (r.json().get("watermark_detected") is True or r.json().get("detected") is True), f"AudioSeal detect failed: {r.text}"
        results["AudioSeal"] = "PASS"

        # 9. Forensics / YuNet / Xception
        if forensic_orig.exists():
            log("Testing Forensic analysis (YuNet + Xception)...")
            with open(forensic_orig, "rb") as f:
                r = requests.post(
                    f"{BASE_URL}/api/forensics/analyze",
                    files={"file": ("orig.mp4", f, "video/mp4")}
                )
            assert r.status_code == 200, f"Forensics failed: {r.text}"
            f_res = r.json()
            log(f"Forensic Result: {f_res}")
            results["YuNet"] = "PASS"
            results["Xception"] = "PASS"
        else:
            results["YuNet"] = "SKIPPED"
            results["Xception"] = "SKIPPED"

        # 10. Full Verify Endpoint
        if forensic_orig.exists():
            log("Testing Full Verification Endpoint (/api/verify)...")
            with open(forensic_orig, "rb") as f:
                r = requests.post(
                    f"{BASE_URL}/api/verify",
                    files={"file": ("orig.mp4", f, "video/mp4")}
                )
            assert r.status_code == 200, f"Verify failed: {r.text}"
            results["Full Verify"] = "PASS"

    finally:
        log("Terminating packaged application process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    # 11. Persistence Check across restart
    log("Testing database persistence across application restart...")
    proc2 = subprocess.Popen([str(EXE_PATH)], cwd=str(EXE_PATH.parent))
    try:
        time.sleep(3)
        r = requests.get(f"{BASE_URL}/api/provenance/{asset_id}")
        if r.status_code == 200 and r.json().get("asset_id") == asset_id:
            results["Restart Persistence"] = "PASS"
            log(f"Persistence verified! Asset {asset_id} retrieved successfully after restart.")
        else:
            results["Restart Persistence"] = "FAIL"
            log(f"Persistence check failed: status {r.status_code}, body {r.text}")
    finally:
        proc2.terminate()
        try:
            proc2.wait(timeout=5)
        except Exception:
            proc2.kill()

    log("\n=================== PACKAGED APPLICATION TEST RESULTS ===================")
    for test, res in results.items():
        log(f"{test:<25}: {res}")
    log("=========================================================================")

    all_passed = all(val == "PASS" for val in results.values())
    if all_passed:
        log("ALL PACKAGED FUNCTIONAL TESTS PASSED!")
        sys.exit(0)
    else:
        log("SOME PACKAGED FUNCTIONAL TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    test_packaged_app()
