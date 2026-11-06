import hashlib
from pathlib import Path
from typing import Dict, Union


def calculate_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Calculates SHA-256 checksum of a file using streaming chunks."""
    path = Path(file_path).resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found for hashing: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest().lower()


def verify_file_hash(file_path: Union[str, Path], expected_sha256: str) -> Dict[str, object]:
    """Verifies a file's actual SHA-256 against an expected hash string."""
    actual_hash = calculate_sha256(file_path)
    clean_expected = expected_sha256.strip().lower()
    matches = actual_hash == clean_expected
    return {
        "matches": matches,
        "expected_sha256": clean_expected,
        "actual_sha256": actual_hash,
    }
