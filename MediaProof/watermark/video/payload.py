import uuid
from typing import Dict, List, Optional

from watermark.video.config import MAGIC_HEADER_BYTES, PAYLOAD_BIT_LENGTH


def bytes_to_bits(data_bytes: bytes) -> List[int]:
    """Converts a bytes object into a list of 0/1 bits."""
    bits = []
    for byte in data_bytes:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """Converts a list of 0/1 bits back to bytes."""
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i : i + 8]
        byte_val = 0
        for bit in chunk:
            byte_val = (byte_val << 1) | (1 if bit > 0.5 else 0)
        byte_array.append(byte_val)
    return bytes(byte_array)


def encode_watermark_token_payload(token_hex: str) -> List[int]:
    """Encodes a hex watermark token (4 or 8 hex chars) into a 256-bit repetition coded binary list."""
    clean_hex = token_hex.strip().upper()
    if len(clean_hex) == 4:
        # 16-bit token -> 16x repetition (16 * 16 = 256 bits)
        val = int(clean_hex, 16)
        bits_16 = [(val >> (15 - i)) & 1 for i in range(16)]
        bits_256 = []
        for b in bits_16:
            bits_256.extend([b] * 16)
        return bits_256

    if len(clean_hex) == 8:
        # 32-bit token -> 8x repetition (32 * 8 = 256 bits)
        val = int(clean_hex, 16)
        bits_32 = [(val >> (31 - i)) & 1 for i in range(32)]
        bits_256 = []
        for b in bits_32:
            bits_256.extend([b] * 8)
        return bits_256

    raise ValueError(f"Watermark token hex string must be 4 or 8 characters, got '{token_hex}'")


def decode_watermark_token_payload(logits_or_bits: List[float], token_bits: int = 16) -> str:
    """Decodes a 256-bit prediction vector back into a hex watermark token via majority voting."""
    if len(logits_or_bits) < PAYLOAD_BIT_LENGTH:
        raise ValueError(f"Input bit array must have at least {PAYLOAD_BIT_LENGTH} values, got {len(logits_or_bits)}")

    if token_bits == 16:
        # 16-bit token from 16x repetition blocks
        res_bits = []
        for i in range(16):
            block = logits_or_bits[i * 16 : (i + 1) * 16]
            vote_sum = sum(1 if val > 0.0 else 0 for val in block)
            res_bits.append(1 if vote_sum >= 8 else 0)

        val = 0
        for b in res_bits:
            val = (val << 1) | b
        return f"{val:04X}"

    if token_bits == 32:
        # 32-bit token from 8x repetition blocks
        res_bits = []
        for i in range(32):
            block = logits_or_bits[i * 8 : (i + 1) * 8]
            vote_sum = sum(1 if val > 0.0 else 0 for val in block)
            res_bits.append(1 if vote_sum >= 4 else 0)

        val = 0
        for b in res_bits:
            val = (val << 1) | b
        return f"{val:08X}"

    raise ValueError(f"Unsupported token_bits size: {token_bits}. Supported: 16, 32")


def encode_asset_id_payload(asset_id: str, sha256_hex: Optional[str] = None) -> List[int]:
    """Encodes a MediaProof Asset ID (MP-uuid) into a 256-bit binary payload."""
    clean_uuid_str = asset_id.replace("MP-", "").strip()
    try:
        raw_uuid_bytes = uuid.UUID(clean_uuid_str).bytes
    except Exception as e:
        raise ValueError(f"Invalid Asset ID UUID string for encoding: '{asset_id}': {e}")

    # Magic Header (2 bytes)
    header = MAGIC_HEADER_BYTES

    # SHA-256 Fragment (4 bytes)
    sha256_bytes = b"\x00\x00\x00\x00"
    if sha256_hex and len(sha256_hex) >= 8:
        try:
            sha256_bytes = bytes.fromhex(sha256_hex[:8])
        except Exception:
            pass

    # Flags / Version (2 bytes)
    flags = b"\x00\x01"

    # Padding (8 bytes)
    padding = b"\x00" * 8

    # Combine 32 bytes (256 bits)
    full_payload = header + raw_uuid_bytes + sha256_bytes + flags + padding
    assert len(full_payload) == 32, f"Payload must be 32 bytes, got {len(full_payload)}"

    bits = bytes_to_bits(full_payload)
    assert len(bits) == PAYLOAD_BIT_LENGTH, f"Bit length must be {PAYLOAD_BIT_LENGTH}"
    return bits


def decode_asset_id_payload(bits: List[int]) -> Dict[str, object]:
    """Decodes a 256-bit binary payload back into a MediaProof Asset ID."""
    if len(bits) < PAYLOAD_BIT_LENGTH:
        return {"valid": False, "asset_id": None, "sha256_fragment": None}

    payload_bytes = bits_to_bytes(bits[:PAYLOAD_BIT_LENGTH])
    header = payload_bytes[0:2]

    # Validate Magic Header
    if header != MAGIC_HEADER_BYTES:
        return {"valid": False, "asset_id": None, "sha256_fragment": None}

    raw_uuid_bytes = payload_bytes[2:18]
    sha256_bytes = payload_bytes[18:22]

    try:
        recovered_uuid = uuid.UUID(bytes=raw_uuid_bytes)
        asset_id = f"MP-{recovered_uuid}"
        sha256_fragment = sha256_bytes.hex().lower()
        return {
            "valid": True,
            "asset_id": asset_id,
            "sha256_fragment": sha256_fragment,
        }
    except Exception:
        return {"valid": False, "asset_id": None, "sha256_fragment": None}
