import hashlib
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from media.ffmpeg import get_ffmpeg_status
from provenance.database import init_db, get_db_path
from provenance.hashing import calculate_sha256, verify_file_hash
from provenance.models import generate_asset_id
from provenance.service import ProvenanceService

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_MEDIA_DIR = PROJECT_ROOT / "data" / "temp" / "test_media"


def is_ffmpeg_available() -> bool:
    status = get_ffmpeg_status()
    return bool(status["ffmpeg_available"] and status["ffprobe_available"])


def test_sha256_known_value(tmp_path):
    test_file = tmp_path / "hello.txt"
    content = b"Hello MediaProof Provenance"
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest().lower()
    actual_hash = calculate_sha256(test_file)

    assert actual_hash == expected_hash
    assert len(actual_hash) == 64


def test_streaming_hashing_and_verification(tmp_path):
    test_file = tmp_path / "large_dummy.bin"
    # Write 200 KB dummy file
    test_file.write_bytes(b"A" * 200000)

    hash1 = calculate_sha256(test_file, chunk_size=4096)
    hash2 = calculate_sha256(test_file, chunk_size=65536)
    assert hash1 == hash2

    ver_res = verify_file_hash(test_file, hash1)
    assert ver_res["matches"] is True
    assert ver_res["actual_sha256"] == hash1


def test_missing_file_hashing():
    fake_path = PROJECT_ROOT / "data" / "temp" / "nonexistent_hash_file.wav"
    with pytest.raises(FileNotFoundError):
        calculate_sha256(fake_path)


def test_database_initialization():
    init_db()
    db_path = get_db_path()
    assert db_path.exists()
    assert db_path.is_file()


def test_asset_id_format():
    asset_id = generate_asset_id()
    assert asset_id.startswith("MP-")
    assert len(asset_id) > 20


def test_register_media_and_retrieval():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_file = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_file.exists():
        pytest.skip("Test combined MP4 file not generated")

    record = ProvenanceService.register_media(mp4_file)
    assert record.asset_id.startswith("MP-")
    assert record.sha256 == calculate_sha256(mp4_file)
    assert record.container_format != "unknown"
    assert record.file_size_bytes > 0
    assert record.media_type in ("video", "audio")

    # Retrieve by Asset ID
    retrieved = ProvenanceService.get_asset(record.asset_id)
    assert retrieved is not None
    assert retrieved.asset_id == record.asset_id
    assert retrieved.sha256 == record.sha256

    # Retrieve by SHA-256
    retrieved_by_hash = ProvenanceService.find_asset_by_sha256(record.sha256)
    assert retrieved_by_hash is not None
    assert retrieved_by_hash.asset_id == record.asset_id


def test_duplicate_registration_handling():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    wav_file = TEST_MEDIA_DIR / "test_audio.wav"
    if not wav_file.exists():
        pytest.skip("Test WAV file not generated")

    rec1 = ProvenanceService.register_media(wav_file)
    rec2 = ProvenanceService.register_media(wav_file)

    assert rec1.asset_id == rec2.asset_id
    assert rec2.already_registered is True


def test_hash_verification_exact_and_mismatch():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_file = TEST_MEDIA_DIR / "test_combined.mp4"
    wav_file = TEST_MEDIA_DIR / "test_audio.wav"

    rec = ProvenanceService.register_media(mp4_file)

    # Test Exact Match
    exact_res = ProvenanceService.verify_asset_hash(mp4_file, rec.asset_id)
    assert exact_res.hash_match is True
    assert exact_res.status == "EXACT_MATCH"
    assert exact_res.actual_sha256 == rec.sha256

    # Test Mismatch (using a different file content against rec.asset_id)
    mismatch_res = ProvenanceService.verify_asset_hash(wav_file, rec.asset_id)
    assert mismatch_res.hash_match is False
    assert mismatch_res.status == "HASH_MISMATCH"


def test_api_provenance_endpoints():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_file = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_file.exists():
        pytest.skip("Test combined MP4 file not generated")

    # 1. API Registration
    with open(mp4_file, "rb") as f:
        response = client.post(
            "/api/provenance/register",
            files={"file": ("uploaded_for_reg.mp4", f, "video/mp4")}
        )

    assert response.status_code in (200, 201)
    reg_data = response.json()
    assert "asset_id" in reg_data
    asset_id = reg_data["asset_id"]
    sha256 = reg_data["sha256"]

    # 2. API Retrieval by Asset ID
    res_get = client.get(f"/api/provenance/{asset_id}")
    assert res_get.status_code == 200
    get_data = res_get.json()
    assert get_data["asset_id"] == asset_id

    # 3. API Retrieval by SHA-256
    res_hash = client.get(f"/api/provenance/hash/{sha256}")
    assert res_hash.status_code == 200
    hash_data = res_hash.json()
    assert hash_data["asset_id"] == asset_id

    # 4. API Hash Verification (Exact Match)
    with open(mp4_file, "rb") as f:
        res_ver = client.post(
            "/api/provenance/verify",
            files={"file": ("uploaded_for_ver.mp4", f, "video/mp4")},
            data={"asset_id": asset_id}
        )

    assert res_ver.status_code == 200
    ver_data = res_ver.json()
    assert ver_data["hash_match"] is True
    assert ver_data["status"] == "EXACT_MATCH"

    # 5. API Hash Verification (Mismatch)
    wav_file = TEST_MEDIA_DIR / "test_audio.wav"
    with open(wav_file, "rb") as f:
        res_ver_mismatch = client.post(
            "/api/provenance/verify",
            files={"file": ("different_file.wav", f, "audio/wav")},
            data={"asset_id": asset_id}
        )

    assert res_ver_mismatch.status_code == 200
    ver_mismatch_data = res_ver_mismatch.json()
    assert ver_mismatch_data["hash_match"] is False
    assert ver_mismatch_data["status"] == "HASH_MISMATCH"


def test_hash_verification_unknown_asset_service_and_api():
    """Regression Test: Unregistered asset ID returns UNKNOWN_ASSET with hash_match=False
    both via ProvenanceService directly and via the HTTP API."""
    mp4_file = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_file.exists():
        pytest.skip("Test combined MP4 file not generated")

    nonexistent_asset_id = "MP-00000000-0000-0000-0000-000000000000"

    # 1. Direct Service Call
    res_svc = ProvenanceService.verify_asset_hash(mp4_file, nonexistent_asset_id)
    assert res_svc.hash_match is False
    assert res_svc.status == "UNKNOWN_ASSET"
    assert res_svc.asset_id == nonexistent_asset_id
    assert res_svc.expected_sha256 == ""

    # 2. HTTP API Endpoint Call
    with open(mp4_file, "rb") as f:
        res_api = client.post(
            "/api/provenance/verify",
            files={"file": ("unknown_asset.mp4", f, "video/mp4")},
            data={"asset_id": nonexistent_asset_id}
        )

    assert res_api.status_code == 200
    api_data = res_api.json()
    assert api_data["hash_match"] is False
    assert api_data["status"] == "UNKNOWN_ASSET"
    assert api_data["expected_sha256"] == ""

