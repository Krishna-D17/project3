# MediaProof — Enhancement Plan: Beyond Watermarking

## 1. Diagnosis — What Is Missing Today

The current system already has **six evidence layers** inside `VerificationResult`, but almost all verification weight falls on:

| Layer | Current Strength | Gap |
|---|---|---|
| **SHA-256 hash** | EXTREME (byte-for-byte) | Breaks on any re-encode; useless for derivatives |
| **Watermark** | STRONG (when intact) | Survives compression but can be stripped; does NOT prove unmodified content |
| **Forensic signals** | LOW-MEDIUM heuristic | FFT, Laplacian, temporal noise — all single-file analysis only; no baseline comparison |
| **Metadata signals** | LOW heuristic | Codec / container checks; no comparison against registered original |
| **ML (deepfake)** | NOT_INSTALLED by default | Requires heavyweight checkpoint |
| **Manipulation** | Maps to ML result | No detector → NONE |

**Core problem:**  
When a watermarked video is returned (after re-encoding, resizing, or partial edit), the watermark may still be present, the SHA-256 will differ, and the forensic signals have nothing to *compare against* — they only inspect the current file in isolation. A partial edit (e.g., 10 seconds deleted) passes the watermark check and has only weak heuristic evidence against it.

---

## 2. Three Pillars We Will Add

```
Registration Time                       Verification Time
─────────────────                       ─────────────────
Original video ingested                 Returned video submitted
         │                                       │
  ┌──────▼──────────────────────────────────────▼──────┐
  │  Pillar A: Perceptual Fingerprint (dHash sequence)  │
  │  • Extract 1 fps, 32×32 grayscale frames via FFmpeg │
  │  • dHash each frame → 64-bit integer                │
  │  • Store ordered list in SQLite per asset            │
  │  • On verify: re-extract & align sequences           │
  │  • Detect: frame deletion, insertion, reorder, splice│
  └─────────────────────────────────────────────────────┘
         │                                       │
  ┌──────▼──────────────────────────────────────▼──────┐
  │  Pillar B: Rich Metadata Snapshot                   │
  │  • Store: duration_s, width, height, fps, frame_cnt │
  │  • Store: video_codec, audio_codec, sample_rate     │
  │  • Store: creation_time tag, encoder tag            │
  │  • On verify: compare incoming vs registered values  │
  │  • Flag: duration drift, resolution change, FPS chg  │
  └─────────────────────────────────────────────────────┘
         │                                       │
  ┌──────▼──────────────────────────────────────▼──────┐
  │  Pillar C: Payload Integrity — SHA-256 fragment     │
  │  • Already partially done in payload.py             │
  │  • Extend: store full 32-byte hash at registration  │
  │  • On detect: compare hash fragment in watermark     │
  │  payload against DB; mismatch = content changed      │
  └─────────────────────────────────────────────────────┘
```

All three pillars are:
- **CPU-only** (FFmpeg + NumPy + stdlib)
- **Zero new ML model weights**
- **Embedded in SQLite** (no new database)
- **Sub-second per check** on low-end hardware (32×32 frames, 1 fps sampling)

---

## 3. Pillar A — Perceptual Fingerprint (Temporal dHash Sequence)

### 3.1 What dHash Is

A **difference hash (dHash)** reduces each frame to a 64-bit integer by:
1. Downsizing frame to 9×8 grayscale pixels via FFmpeg
2. Computing left-to-right pixel gradient (8 comparisons × 8 rows = 64 bits)
3. Encoding result as a single 64-bit integer

Two dHash values that differ by **≤ 5 bits** (Hamming distance) represent visually near-identical frames. This survives:
- H.264 → H.265 re-encode
- Minor brightness/contrast adjustments
- Slight scaling / letterboxing

### 3.2 What Gets Stored at Registration

```
Table: asset_fingerprints
  asset_id         TEXT   (FK → assets.asset_id)
  frame_index      INTEGER
  dhash_int        INTEGER  (64-bit signed, stores dHash as integer)
  PRIMARY KEY (asset_id, frame_index)
```

Estimated storage: ~8 bytes × N frames. For a 60-second video sampled at 1 fps → 60 rows × 8 bytes = **480 bytes** — negligible.

### 3.3 What Happens at Verification

1. **Extract** the returning video's dHash sequence (same 32×32 / 1 fps FFmpeg pipe)
2. **Load** registered sequence from `asset_fingerprints`
3. **Align** sequences with a simple sequential sliding-window matcher (no DTW needed at this scale)
4. **Compute** integrity metrics:
   - `frame_count_match`: registered N vs actual N
   - `sequence_match_ratio`: fraction of frames matched within Hamming ≤ 5
   - `max_contiguous_anomaly`: longest run of unmatched frames
   - `tamper_type`: `NONE` / `FRAME_DELETION` / `FRAME_INSERTION` / `CONTENT_SPLICE`
5. **Emit** as a new `fingerprint_evidence: EvidenceLayer` in `VerificationResult`

### 3.4 Tamper Classification Rules

| Condition | Tamper Type | Risk Level |
|---|---|---|
| `seq_match_ratio ≥ 0.95` | NONE | LOW |
| Frame count differs >5% AND `seq_match_ratio < 0.80` | FRAME_DELETION or INSERTION | MEDIUM |
| `max_contiguous_anomaly ≥ 30 frames` (at 1 fps = 30s of unrecognised content) | CONTENT_SPLICE | HIGH |
| `seq_match_ratio < 0.50` | SEVERE_MANIPULATION | HIGH |

### 3.5 New Module: `fingerprint/`

```
fingerprint/
  __init__.py
  models.py          # FingerprintRecord dataclass
  database.py        # CREATE TABLE asset_fingerprints
  repository.py      # insert/get fingerprint sequences
  extractor.py       # FFmpeg pipe → dHash sequence (CPU-only)
  comparator.py      # sequence alignment + tamper classification
  service.py         # FingerprintService (register / compare)
```

---

## 4. Pillar B — Rich Metadata Snapshot & Comparison

### 4.1 Problem with Current Metadata

`AssetRecord` already stores: `duration_seconds`, `width`, `height`, `fps`, `video_codec`, `audio_codec`, `sample_rate`, `channels`.

But **verification never compares** these against the returning file — they are only displayed in the provenance record. A clip that has been trimmed from 60s → 40s still passes verification.

### 4.2 New Metadata Comparison Signal

At verification time (after `prov_record` is found):

```python
metadata_drift = compare_metadata(
    registered=prov_record,
    current=probe_res
)
```

**Checks to run:**

| Field | Tolerance | Violation |
|---|---|---|
| `duration_seconds` | ± 1.0 s | Duration trimmed / extended |
| `width` × `height` | Exact | Resolution changed (crop, scale) |
| `fps` | ± 0.1 | Frame rate changed |
| `video_codec` | Exact | Re-encoded to different codec |
| `audio_codec` | Exact | Audio track re-encoded |
| `sample_rate` | Exact | Audio sample rate changed |
| `channels` | Exact | Audio channels changed |

**Result:** New `metadata_drift_evidence` layer (or merge into `metadata_evidence` with stronger scoring).

### 4.3 Extended Metadata Fields to Store

Add to `assets` table (via migration):
```sql
ALTER TABLE assets ADD COLUMN frame_count INTEGER;
ALTER TABLE assets ADD COLUMN creation_time_tag TEXT;
ALTER TABLE assets ADD COLUMN encoder_tag TEXT;
ALTER TABLE assets ADD COLUMN bit_rate INTEGER;
```

These are extracted from FFprobe tags at registration time and compared at verify time.

---

## 5. Pillar C — Watermark Payload Hash Fragment Integrity

### 5.1 Current State

In `payload.py`, `encode_asset_id_payload()` already encodes the first 4 bytes of the SHA-256 hash into bytes 18–21 of the 256-bit payload. `decode_asset_id_payload()` returns `sha256_fragment`.

However: **this fragment is never validated against the database during verification**.

### 5.2 What to Add

In `VerificationResult` / detection flow:

```python
# After watermark detection returns sha256_fragment:
if wm_sha256_fragment and prov_record:
    stored_sha256_prefix = prov_record.sha256[:8]   # first 4 bytes = 8 hex chars
    fragment_match = (wm_sha256_fragment == stored_sha256_prefix)
    # fragment_match=False means: watermark is from original but content was modified
```

**Evidence interpretation:**
- `fragment_match = True` → Watermark payload is consistent with registered original hash → supports authenticity
- `fragment_match = False` + watermark resolved → Video content was modified (hash changed) but watermark survived → Strong signal of derivative manipulation
- `fragment_match = None` → Fragment extraction failed or no provenance record

This gives a **within-watermark tamper indicator** that requires zero extra computation — the data is already embedded.

---

## 6. Integration into VerificationResult

Add two new `EvidenceLayer` fields to `VerificationResult`:

```python
@dataclass
class VerificationResult:
    ...
    fingerprint_evidence: EvidenceLayer      # NEW — Pillar A
    hash_fragment_evidence: EvidenceLayer    # NEW — Pillar C
    metadata_drift_evidence: EvidenceLayer   # UPGRADED — Pillar B
```

Update decision matrix:

| Condition Added | Effect |
|---|---|
| `fingerprint_evidence.strength == HIGH` (severe splice) | Escalates `VERIFIED DERIVATIVE` → `POTENTIALLY MANIPULATED` |
| `hash_fragment_evidence.status == MISMATCH` + watermark present | Adds `MEDIUM` confidence that content changed post-watermark |
| `metadata_drift_evidence` duration mismatch > 5s | Adds limitation note; escalates if combined with fingerprint anomaly |

---

## 7. Database Schema Changes

### 7.1 New Table: `asset_fingerprints`

```sql
CREATE TABLE IF NOT EXISTS asset_fingerprints (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id     TEXT NOT NULL,
    frame_index  INTEGER NOT NULL,
    dhash_int    INTEGER NOT NULL,
    UNIQUE(asset_id, frame_index),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_fingerprints_asset_id ON asset_fingerprints(asset_id);
```

### 7.2 Extend `assets` Table

```sql
ALTER TABLE assets ADD COLUMN frame_count    INTEGER;
ALTER TABLE assets ADD COLUMN creation_time_tag TEXT;
ALTER TABLE assets ADD COLUMN encoder_tag    TEXT;
ALTER TABLE assets ADD COLUMN bit_rate       INTEGER;
```

These columns are optional (`NULL` for old records) — fully backward-compatible.

---

## 8. File-by-File Implementation Map

| File | Change Type | Description |
|---|---|---|
| `fingerprint/__init__.py` | **NEW** | Module exports |
| `fingerprint/models.py` | **NEW** | `FingerprintRecord`, `FingerprintComparisonResult` dataclasses |
| `fingerprint/database.py` | **NEW** | `create_fingerprint_table()`, schema migration |
| `fingerprint/repository.py` | **NEW** | `insert_fingerprint_sequence()`, `get_fingerprint_sequence()` |
| `fingerprint/extractor.py` | **NEW** | `extract_dhash_sequence(file_path, fps=1)` via FFmpeg pipe |
| `fingerprint/comparator.py` | **NEW** | `compare_sequences()` → sliding-window Hamming alignment |
| `fingerprint/service.py` | **NEW** | `FingerprintService.register()`, `FingerprintService.compare()` |
| `provenance/database.py` | **EXTEND** | Add schema migration for new columns on `assets` |
| `provenance/models.py` | **EXTEND** | Add `frame_count`, `creation_time_tag`, `encoder_tag`, `bit_rate` fields |
| `provenance/repository.py` | **EXTEND** | Update `insert_asset()` and `row_to_asset_record()` for new fields |
| `provenance/service.py` | **EXTEND** | Capture new fields during `register_media()` |
| `provenance/verification.py` | **EXTEND** | Add fingerprint, hash-fragment, metadata-drift evidence layers; update decision matrix |
| `backend/main.py` | **EXTEND** | Include `fingerprint_evidence` and `hash_fragment_evidence` in `/api/verify` response |
| `provenance/database.py` | **EXTEND** | Run `CREATE TABLE asset_fingerprints` on startup |

---

## 9. Performance Budget (Low-End Device)

Target: CPU-only, ≤ 60 seconds for a 5-minute video registration.

| Step | Cost Estimate |
|---|---|
| Extract 1 fps, 32×32 grayscale via FFmpeg pipe | ~0.5–1s per minute of video |
| dHash 64-bit integer per frame (pure Python / NumPy) | ~0.1 ms per frame |
| SQLite INSERT of 300 rows (5 min video) | < 10 ms |
| **Total per 5-min video** | **~3–6 seconds** |

At verification:
| Step | Cost Estimate |
|---|---|
| Re-extract dHash sequence | Same as above |
| Load registered sequence from SQLite | < 5 ms |
| Sliding-window alignment (300 frames) | < 50 ms pure NumPy |
| **Total verification overhead** | **~3–6 seconds** |

---

## 10. Phased Delivery Plan

### Phase 1 — Foundation (Pillar A: Fingerprint Module) ✦ Priority HIGH
1. Create `fingerprint/` module with all subfiles
2. Extend `provenance/database.py` to create `asset_fingerprints` table on startup
3. Call `FingerprintService.register()` inside `ProvenanceService.register_media()`
4. Wire `FingerprintService.compare()` into `VerificationService.verify_media()`
5. Add `fingerprint_evidence: EvidenceLayer` to `VerificationResult`
6. Update decision matrix to escalate on HIGH fingerprint evidence
7. Add tests for: registration, exact match, frame deletion, splice detection

### Phase 2 — Metadata Drift (Pillar B) ✦ Priority MEDIUM
1. Add new columns to `assets` table (migration-safe ALTERs)
2. Extend `AssetRecord` model and `ProvenanceService.register_media()` to capture them
3. Add `compare_metadata()` in a new `fingerprint/metadata_drift.py`
4. Wire into `VerificationService` as `metadata_drift_evidence`
5. Update tests

### Phase 3 — Hash Fragment Validation (Pillar C) ✦ Priority MEDIUM
1. Extract `sha256_fragment` from watermark detection result in `VerificationService`
2. Compare against `prov_record.sha256[:8]`
3. Emit `hash_fragment_evidence` EvidenceLayer
4. Update decision matrix: fragment mismatch + watermark present → add to `conflicts`
5. Add tests

### Phase 4 — API Response & UI ✦ Priority LOW
1. Expose new evidence layers in `/api/verify` JSON response
2. Update frontend verification result card to show fingerprint integrity and metadata drift panels
3. Add a "Fingerprint Details" section showing: frame_count_registered, frame_count_current, match_ratio, tamper_type

---

## 11. Limitations & Scope

| Limitation | Explanation |
|---|---|
| dHash does not detect **colour-only edits** | Colour grading or filter changes will NOT trigger tamper detection (dHash is grayscale) |
| Aggressive **crop + reframe** may still match | If 90% of pixels remain, match ratio stays high |
| **Deepfake face-swap** may not be caught | If temporal content stays plausible, dHash sequence matches; ML model is still the right tool here |
| **Slow motion / speed change** detection requires DTW | Simple sliding window may struggle with >10% speed change; acceptable for Phase 1 |
| **Audio fingerprinting** not covered here | Separate future pillar using audio spectral hashing |

---

## 12. Summary Decision

> **Watermarking alone proves provenance linkage, not content integrity.**  
> The three new pillars convert MediaProof from a "who made it" system to a "who made it AND was it changed" system — without any ML models, cloud services, or GPU dependencies.

The implementation is designed to be **additive**: every new layer degrades gracefully when data is absent (old assets without fingerprints simply receive `INSUFFICIENT` evidence strength, not errors).
