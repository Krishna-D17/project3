import os
import shutil
import uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from forensics.service import ForensicService
from media.ffmpeg import FFmpegExecutionError, FFmpegNotFoundError, FFmpegTimeoutError
from media.service import MediaService
from provenance.service import ProvenanceService
from provenance.verification import VerificationService
from watermark.audio.detect import detect_audio_watermark
from watermark.audio.embed import embed_audio_watermark
from watermark.video.detect import detect_video_watermark
from watermark.video.embed import embed_video_watermark

app = FastAPI(
    title="MediaProof",
    description="Local CPU-only media provenance, invisible watermarking, and forensic decision engine backend",
    version="0.1.0"
)

from backend.paths import get_frontend_dir, get_protected_data_dir, get_temp_data_dir

TEMP_DIR = get_temp_data_dir()
PROTECTED_DIR = get_protected_data_dir()
FRONTEND_DIR = get_frontend_dir()

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def read_root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "ok",
        "message": "MediaProof backend is running",
        "application": "MediaProof",
        "version": "0.1.0"
    }


@app.get("/api/media/protected/{filename}")
def get_protected_media(filename: str):
    """Serves protected watermarked media files from data/protected for local download or streaming."""
    safe_name = Path(filename).name
    file_path = PROTECTED_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Protected file '{safe_name}' not found."
        )
    return FileResponse(file_path, filename=safe_name)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "application": "MediaProof",
        "version": "0.1.0",
        "mode": "cpu"
    }


@app.get("/api/media/status")
def media_status():
    """Reports availability and executable paths for FFmpeg and FFprobe."""
    return MediaService.get_engine_status()


@app.post("/api/media/probe")
async def probe_media(file: UploadFile = File(...)):
    """Accepts a media file upload, probes it via FFprobe, returns metadata, and cleans temp storage."""
    status_info = MediaService.get_engine_status()
    if not status_info.get("ffprobe_available"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FFmpeg/FFprobe binaries are not available on backend server."
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename."
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_filename = f"upload_{uuid.uuid4().hex}_{Path(file.filename).name}"
    temp_path = TEMP_DIR / temp_filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if temp_path.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )

        result = MediaService.probe_file(temp_path)
        result_dict = result.to_dict()
        result_dict["filename"] = file.filename
        result_dict["path"] = file.filename

        return JSONResponse(status_code=200, content=result_dict)

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File error: {str(e)}"
        )
    except FFmpegTimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Media probing timed out: {str(e)}"
        )
    except (FFmpegExecutionError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid, corrupt, or unsupported media file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected media processing error: {str(e)}"
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.post("/api/provenance/register")
async def register_provenance(
    file: UploadFile = File(...),
    parent_asset_id: Optional[str] = Form(None),
    relationship_type: str = Form("ORIGINAL"),
):
    """Uploads media, probes properties, hashes SHA-256, registers asset provenance in SQLite, and cleans temp storage."""
    status_info = MediaService.get_engine_status()
    if not status_info.get("ffprobe_available"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FFmpeg/FFprobe binaries are not available on backend server."
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename."
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_filename = f"upload_reg_{uuid.uuid4().hex}_{Path(file.filename).name}"
    temp_path = TEMP_DIR / temp_filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if temp_path.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )

        record = ProvenanceService.register_media(
            temp_path,
            parent_asset_id=parent_asset_id,
            relationship_type=relationship_type,
        )
        record_dict = record.to_dict()
        record_dict["original_filename"] = file.filename
        record_dict["original_path"] = file.filename

        res_status = status.HTTP_200_OK if record.already_registered else status.HTTP_201_CREATED
        return JSONResponse(status_code=res_status, content=record_dict)

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to register media provenance: {str(e)}"
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.get("/api/provenance/{asset_id}")
def get_provenance(asset_id: str):
    """Retrieves a provenance record by Asset ID."""
    record = ProvenanceService.get_asset(asset_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset record not found for Asset ID: {asset_id}"
        )
    return record.to_dict()


@app.get("/api/provenance/hash/{sha256}")
def get_provenance_by_hash(sha256: str):
    """Retrieves a provenance record by SHA-256 hash string."""
    record = ProvenanceService.find_asset_by_sha256(sha256)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset record not found for SHA-256: {sha256}"
        )
    return record.to_dict()


@app.post("/api/provenance/verify")
async def verify_provenance(
    file: UploadFile = File(...),
    asset_id: str = Form(...),
):
    """Uploads media file and verifies SHA-256 hash against a registered Asset ID record."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename."
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_filename = f"upload_ver_{uuid.uuid4().hex}_{Path(file.filename).name}"
    temp_path = TEMP_DIR / temp_filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if temp_path.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )

        verification = ProvenanceService.verify_asset_hash(temp_path, asset_id)
        res_dict = verification.to_dict()
        res_dict["filename"] = file.filename
        return JSONResponse(status_code=status.HTTP_200_OK, content=res_dict)

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification processing error: {str(e)}"
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.post("/api/watermark/video/embed")
async def embed_video_wm_endpoint(
    file: UploadFile = File(...),
    asset_id: str = Form(...),
    engine: str = Form("custom"),
):
    """Embeds an invisible Asset ID watermark into an uploaded video file via selected engine ('custom' or 'videoseal')."""
    status_info = MediaService.get_engine_status()
    if not status_info.get("ffmpeg_available"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FFmpeg/FFprobe binaries are not available on backend server."
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename."
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    PROTECTED_DIR.mkdir(parents=True, exist_ok=True)

    temp_in = TEMP_DIR / f"upload_wm_in_{uuid.uuid4().hex}_{Path(file.filename).name}"
    protected_out = PROTECTED_DIR / f"wm_{Path(file.filename).name}"

    try:
        with open(temp_in, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        res = embed_video_watermark(temp_in, protected_out, asset_id, engine_type=engine)
        return JSONResponse(status_code=status.HTTP_200_OK, content=res)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid video file or engine option: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video watermarking failed: {str(e)}"
        )
    finally:
        if temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass


@app.post("/api/watermark/video/detect")
async def detect_video_wm_endpoint(
    file: UploadFile = File(...),
    engine: str = Form("custom"),
):
    """Detects and extracts invisible MediaProof watermarks from an uploaded video file via selected engine."""
    status_info = MediaService.get_engine_status()
    if not status_info.get("ffprobe_available"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FFmpeg/FFprobe binaries are not available on backend server."
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename."
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_in = TEMP_DIR / f"upload_wm_det_{uuid.uuid4().hex}_{Path(file.filename).name}"

    try:
        with open(temp_in, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        res = detect_video_watermark(temp_in, engine_type=engine)
        return JSONResponse(status_code=status.HTTP_200_OK, content=res)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Engine error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video watermark detection error: {str(e)}"
        )
    finally:
        if temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass


@app.post("/api/watermark/audio/embed")
async def embed_audio_wm_endpoint(
    file: UploadFile = File(...),
    asset_id: str = Form(...),
    engine: str = Form("custom"),
):
    """Embeds an invisible Asset ID watermark into an uploaded audio file via selected engine ('custom' or 'audioseal')."""
    status_info = MediaService.get_engine_status()
    if not status_info.get("ffmpeg_available"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FFmpeg/FFprobe binaries are not available on backend server."
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename."
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    PROTECTED_DIR.mkdir(parents=True, exist_ok=True)

    temp_in = TEMP_DIR / f"upload_audio_in_{uuid.uuid4().hex}_{Path(file.filename).name}"
    protected_out = PROTECTED_DIR / f"wm_audio_{Path(file.filename).name}"

    try:
        with open(temp_in, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        res = embed_audio_watermark(temp_in, protected_out, asset_id, engine_type=engine)
        return JSONResponse(status_code=status.HTTP_200_OK, content=res)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audio file or engine option: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio watermarking failed: {str(e)}"
        )
    finally:
        if temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass


@app.post("/api/watermark/audio/detect")
async def detect_audio_wm_endpoint(
    file: UploadFile = File(...),
    engine: str = Form("custom"),
):
    """Detects and extracts invisible MediaProof watermarks from an uploaded audio file via selected engine ('custom' or 'audioseal')."""
    status_info = MediaService.get_engine_status()
    if not status_info.get("ffprobe_available"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FFmpeg/FFprobe binaries are not available on backend server."
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename."
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_in = TEMP_DIR / f"upload_audio_det_{uuid.uuid4().hex}_{Path(file.filename).name}"

    try:
        with open(temp_in, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        res = detect_audio_watermark(temp_in, engine_type=engine)
        return JSONResponse(status_code=status.HTTP_200_OK, content=res)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Engine error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio watermark detection error: {str(e)}"
        )
    finally:
        if temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass


@app.post("/api/forensics/analyze")
async def analyze_forensics_endpoint(
    file: UploadFile = File(...),
):
    """Accepts media file upload, executes multi-modal forensic analysis (metadata, spatial/temporal, acoustic, optional ML), returns structured forensic report."""
    status_info = MediaService.get_engine_status()
    if not status_info.get("ffprobe_available"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FFmpeg/FFprobe binaries are not available on backend server."
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename."
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_in = TEMP_DIR / f"upload_forensics_{uuid.uuid4().hex}_{Path(file.filename).name}"

    try:
        with open(temp_in, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        probe_res = MediaService.probe_file(temp_in)
        m_type = "video" if probe_res.has_video else ("audio" if probe_res.has_audio else "unknown")

        forensic_res = ForensicService.analyze_media(temp_in)
        response_dict = {
            "status": "ok",
            "media_type": m_type,
            "forensic": forensic_res.to_dict(),
            "processing_time_seconds": forensic_res.processing_time_seconds,
        }
        return JSONResponse(status_code=status.HTTP_200_OK, content=response_dict)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forensic analysis failed: {str(e)}"
        )
    finally:
        if temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass


@app.post("/api/verify")
async def unified_verification_endpoint(
    file: UploadFile = File(...),
    video_engine: str = Form("custom"),
    audio_engine: str = Form("custom"),
):
    """Unified Verification API Endpoint: orchestrates cryptographic SHA-256 identity, watermark detection, SQLite provenance lookup, and multi-modal forensic decision matrix."""
    status_info = MediaService.get_engine_status()
    if not status_info.get("ffprobe_available"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FFmpeg/FFprobe binaries are not available on backend server."
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename."
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_in = TEMP_DIR / f"upload_verify_{uuid.uuid4().hex}_{Path(file.filename).name}"

    try:
        with open(temp_in, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        verify_res = VerificationService.verify_media(
            file_path=temp_in,
            video_engine=video_engine,
            audio_engine=audio_engine,
        )
        res_dict = verify_res.to_dict()
        if verify_res.status == "error":
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=res_dict)
        return JSONResponse(status_code=status.HTTP_200_OK, content=res_dict)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification decision engine failed: {str(e)}"
        )
    finally:
        if temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass
