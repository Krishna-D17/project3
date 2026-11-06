import datetime
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import torch

def download_and_inspect():
    print("======================================================================")
    print("TASK 1 & 2: DOWNLOADING & VERIFYING OFFICIAL DEEPFAKEBENCH XCEPTION")
    print("======================================================================")

    url = "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth"
    project_root = Path(__file__).resolve().parent.parent
    audit_dir = project_root / "data" / "temp" / "audit_models"
    audit_dir.mkdir(parents=True, exist_ok=True)
    target_path = audit_dir / "xception_best.pth"

    download_start = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print(f"Download URL: {url}")
    print(f"Audit Target Path: {target_path}")

    if not target_path.exists():
        print("Downloading official checkpoint...")
        # Use urllib with standard User-Agent header to support GitHub redirect download
        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
            shutil_copy = response.read()
            out_file.write(shutil_copy)
        print("Download complete.")
    else:
        print("Checkpoint file already exists in audit location.")

    file_size = target_path.stat().st_size
    print(f"File Size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")

    # Calculate SHA-256 Hash
    print("Calculating SHA-256 hash...")
    sha256_hash = hashlib.sha256()
    with open(target_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256_hash.update(chunk)
    sha256_hex = sha256_hash.hexdigest()
    print(f"SHA-256 Hash: {sha256_hex}")

    timestamp_now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # TASK 3: Inspect Checkpoint Structure
    print("\n======================================================================")
    print("TASK 3: INSPECTING CHECKPOINT STRUCTURE")
    print("======================================================================")

    try:
        # Load safely on CPU
        ckpt = torch.load(target_path, map_location="cpu")
        ckpt_type = type(ckpt).__name__
        print(f"Checkpoint Loaded Object Type: {ckpt_type}")

        top_level_keys = []
        state_dict = None

        if isinstance(ckpt, dict):
            top_level_keys = list(ckpt.keys())
            print(f"Top-level Dictionary Keys ({len(top_level_keys)}): {top_level_keys}")
            
            # Identify state_dict inside keys if wrapped
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
                print("Found 'state_dict' key in checkpoint dictionary.")
            elif "model" in ckpt:
                state_dict = ckpt["model"]
                print("Found 'model' key in checkpoint dictionary.")
            else:
                # Check if dictionary itself is a state_dict
                sample_val = next(iter(ckpt.values()))
                if isinstance(sample_val, torch.Tensor):
                    state_dict = ckpt
                    print("Checkpoint dictionary IS a direct PyTorch state_dict.")
        elif isinstance(ckpt, torch.nn.Module):
            state_dict = ckpt.state_dict()
            print("Checkpoint is a complete serialized PyTorch Module object.")
        else:
            print(f"Unknown checkpoint format: {ckpt_type}")

        param_count = 0
        first_param_names = []
        classifier_keys = []

        if state_dict is not None:
            param_count = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            all_keys = list(state_dict.keys())
            first_param_names = all_keys[:10]
            
            print(f"Total Parameters Count: {param_count:,}")
            print("First 10 Parameter Names:")
            for k in first_param_names:
                print(f"  - {k} (shape: {list(state_dict[k].shape)})")

            # Identify classifier / linear head layers
            classifier_keys = [k for k in all_keys if any(term in k for term in ["fc", "classifier", "head", "linear", "last_linear"])]
            print("\nClassifier / Final Layer Keys:")
            for k in classifier_keys:
                print(f"  - {k} (shape: {list(state_dict[k].shape)})")

        inspection_report = {
            "download_url": url,
            "filename": target_path.name,
            "file_size_bytes": file_size,
            "sha256": sha256_hex,
            "timestamp": timestamp_now,
            "checkpoint_type": ckpt_type,
            "top_level_keys": top_level_keys,
            "has_state_dict": state_dict is not None,
            "param_count": param_count,
            "first_param_names": first_param_names,
            "classifier_keys": classifier_keys,
        }

        out_report_path = audit_dir / "inspection_report.json"
        with open(out_report_path, "w") as f:
            json.dump(inspection_report, f, indent=2)
        print(f"\nInspection details saved to {out_report_path}")

    except Exception as e:
        print(f"Error inspecting checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    download_and_inspect()
