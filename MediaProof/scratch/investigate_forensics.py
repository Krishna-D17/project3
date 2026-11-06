import json
import os
import shutil
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

from forensics.service import ForensicService
from media.ffmpeg import get_ffmpeg_path, run_ffmpeg_command
from media.service import MediaService
from provenance.hashing import calculate_sha256
from provenance.repository import get_asset_by_sha256, list_assets
from provenance.service import ProvenanceService
from provenance.verification import VerificationService
from watermark.video.embed import embed_video_watermark


def run_investigation():
    print("======================================================================")
    print("PHASE 1-14: FORENSIC FALSE-POSITIVE INVESTIGATION & AUDIT")
    print("======================================================================")

    project_root = Path(__file__).resolve().parent.parent
    orig_path = project_root / "data" / "originals" / "000.mp4"
    wm_videoseal_path = project_root / "data" / "protected" / "wm_000.mp4"
    temp_dir = project_root / "data" / "temp" / "audit_investigation"
    temp_dir.mkdir(parents=True, exist_ok=True)

    if not orig_path.exists():
        print(f"Error: Original video not found at {orig_path}")
        return

    if not wm_videoseal_path.exists():
        print(f"Error: VideoSeal video not found at {wm_videoseal_path}")
        return

    ffmpeg_exe = get_ffmpeg_path()

    # Register original if not registered
    orig_sha = calculate_sha256(orig_path)
    orig_record = get_asset_by_sha256(orig_sha)
    if not orig_record:
        print("Registering original 000.mp4 in SQLite registry...")
        orig_record = ProvenanceService.register_media(orig_path)
    orig_asset_id = orig_record.asset_id
    print(f"Original Asset ID: {orig_asset_id} (SHA-256: {orig_sha[:16]}...)")

    # 1. Create Controlled Intermediate Test Files (Phase 6):
    # A = Original (000.mp4)
    file_a = orig_path

    # B = FFmpeg Re-encode only (no watermark)
    file_b = temp_dir / "000_ffmpeg_reencode.mp4"
    if not file_b.exists():
        print("Generating File B: FFmpeg re-encode only...")
        cmd_b = [
            str(ffmpeg_exe), "-y",
            "-i", str(file_a),
            "-c:v", "libx264", "-crf", "23", "-preset", "medium",
            "-c:a", "aac", "-b:a", "128k",
            str(file_b)
        ]
        run_ffmpeg_command(cmd_b)

    # C = VideoSeal protected (wm_000.mp4)
    file_d = wm_videoseal_path

    # Custom protected video
    file_custom = temp_dir / "000_custom_wm.mp4"
    if not file_custom.exists():
        print("Generating Custom-protected video...")
        try:
            embed_video_watermark(file_a, file_custom, orig_asset_id, engine_type="custom")
        except Exception as e:
            print(f"Warning embedding custom watermark: {e}")

    # E = Controlled Deepfake / Manipulated video (simulated face blur / temporal distortion)
    file_e = temp_dir / "000_deepfake_manipulated.mp4"
    if not file_e.exists():
        print("Generating File E: Simulated deepfake / spatial-temporal manipulation...")
        # Apply heavy spatial blur + temporal frame jitter + noise box to simulate deepfake artifacts
        cmd_e = [
            str(ffmpeg_exe), "-y",
            "-i", str(file_d if file_d.exists() else file_a),
            "-vf", "boxblur=2:1,drawbox=x=50:y=50:w=100:h=100:color=black@0.5:t=fill",
            "-c:v", "libx264", "-crf", "28",
            "-c:a", "aac", "-b:a", "96k",
            str(file_e)
        ]
        run_ffmpeg_command(cmd_e)

    target_files = {
        "A_Original": file_a,
        "B_FFmpeg_Reencode": file_b,
        "C_Custom_Protected": file_custom if file_custom.exists() else file_b,
        "D_VideoSeal_Protected": file_d,
        "E_Manipulated_Deepfake": file_e,
    }

    # Data collection structure for JSON
    audit_data = {}

    print("\n----------------------------------------------------------------------")
    print("PROBING & FORENSIC DECOMPOSITION")
    print("----------------------------------------------------------------------")

    for key, fpath in target_files.items():
        if not fpath.exists():
            continue

        probe = MediaService.probe_file(fpath)
        sha256_val = calculate_sha256(fpath)
        file_size = fpath.stat().st_size

        v_codec = probe.video_streams[0].codec if probe.video_streams else "None"
        v_res = f"{probe.video_streams[0].width}x{probe.video_streams[0].height}" if probe.video_streams else "None"
        v_fps = probe.video_streams[0].fps if probe.video_streams else 0.0
        a_codec = probe.audio_streams[0].codec if probe.audio_streams else "None"

        # Forensic Analysis
        forensic_res = ForensicService.analyze_media(fpath)

        # Verification Service
        verify_res = VerificationService.verify_media(fpath, video_engine="videoseal", audio_engine="audioseal")

        # Decompose Signals
        all_signals = forensic_res.metadata_signals + forensic_res.video_signals + forensic_res.audio_signals
        high_signals = [s for s in all_signals if s.risk_level == "HIGH"]
        med_signals = [s for s in all_signals if s.risk_level == "MEDIUM"]
        low_signals = [s for s in all_signals if s.risk_level == "LOW"]

        raw_calc = (len(high_signals) * 0.40) + (len(med_signals) * 0.15)
        computed_score = min(max(raw_calc, 0.0), 1.0)

        file_info = {
            "key": key,
            "path": str(fpath),
            "sha256": sha256_val,
            "file_size_bytes": file_size,
            "duration_seconds": probe.duration_seconds,
            "container": probe.container_format,
            "video_codec": v_codec,
            "resolution": v_res,
            "fps": v_fps,
            "audio_codec": a_codec,
            "verification": {
                "classification": verify_res.classification,
                "confidence": verify_res.confidence,
                "confidence_basis": verify_res.confidence_basis,
                "hash_status": verify_res.hash_evidence.status,
                "watermark_status": verify_res.watermark_evidence.status,
                "watermark_engine": verify_res.watermark_evidence.details.get("engine_used"),
                "resolved_asset_id": verify_res.watermark_evidence.details.get("resolved_asset_id"),
                "forensic_status": verify_res.forensic_evidence.status,
                "conflicts": verify_res.conflicts,
            },
            "forensics": {
                "raw_status": forensic_res.forensic_status,
                "risk_score": forensic_res.risk_score,
                "generic_forensic_status": forensic_res.generic_forensic_status,
                "manipulation_forensic_status": forensic_res.manipulation_forensic_status,
                "transformation_consistency": forensic_res.transformation_consistency.to_dict(),
                "computed_score": computed_score,
                "high_risk_count": len(high_signals),
                "medium_risk_count": len(med_signals),
                "low_risk_count": len(low_signals),
                "decomposition_formula": f"({len(high_signals)} * 0.40) + ({len(med_signals)} * 0.15) = {computed_score:.2f}",
                "signals": [
                    {
                        "name": s.signal_name,
                        "category": s.category,
                        "risk_level": s.risk_level,
                        "score": s.score,
                        "details": s.details,
                        "evidence_strength": s.evidence_strength,
                        "type": s.type,
                        "score_contribution": 0.40 if s.risk_level == "HIGH" else (0.15 if s.risk_level == "MEDIUM" else 0.0)
                    }
                    for s in all_signals
                ]
            }
        }

        audit_data[key] = file_info

        print(f"\n[{key}] {fpath.name}")
        print(f"  SHA-256: {sha256_val[:16]}... | Size: {file_size} bytes | Duration: {probe.duration_seconds:.2f}s | Resolution: {v_res} @ {v_fps:.2f}fps")
        print(f"  Classification: {verify_res.classification} ({verify_res.confidence} confidence)")
        print(f"  Hash Evidence: {verify_res.hash_evidence.status} | Watermark: {verify_res.watermark_evidence.status} ({verify_res.watermark_evidence.details.get('engine_used')})")
        print(f"  Forensic Risk Score: {forensic_res.risk_score:.2f} ({forensic_res.forensic_status})")
        print(f"  Score Decomposition: {file_info['forensics']['decomposition_formula']}")
        print("  Signal Breakdown:")
        for s in file_info["forensics"]["signals"]:
            marker = " [ANOMALOUS]" if s["risk_level"] in ["MEDIUM", "HIGH"] else ""
            print(f"    - {s['name']}: risk={s['risk_level']} (score={s['score']:.4f}) -> contribution={s['score_contribution']:.2f}{marker}")
            print(f"      Details: {s['details']}")

    # Save to scratch/forensic_comparison.json
    out_json = project_root / "scratch" / "forensic_comparison.json"
    with open(out_json, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nRaw measurements saved to {out_json}")


if __name__ == "__main__":
    run_investigation()
