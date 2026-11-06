# MediaProof --- Comprehensive End-to-End Upgrade Plan

**Document type:** Engineering implementation plan\
**Target:** Local Windows deployment, CPU-only devices\
**Status:** Proposed; each phase requires review and validation before
implementation

## 1. Executive Summary

MediaProof combines asset registration and file hashing, watermark
detection, forensic signals, optional machine-learning inference, and
provenance classification. The proposed upgrade strengthens verification
by combining five independent evidence layers:

1.  Temporal perceptual fingerprints for likely cuts, insertions,
    deletions, and scene reordering.
2.  Technical metadata drift analysis to compare codecs, duration,
    dimensions, frame rate, audio streams, and other properties.
3.  Watermark/content cross-validation to distinguish asset linkage from
    proof of content integrity.
4.  Lightweight CPU-compatible forensic signals and carefully evaluated
    ML models.
5.  A tamper-evident evidence ledger using hash chaining, Ed25519
    signatures, and optional Merkle proofs.

The system must preserve the five existing final classifications:
**Verified Original**, **Verified Derivative**, **Potentially
Manipulated**, **Unknown Provenance**, and **Insufficient Evidence**.

> Performance, memory, model-size, and accuracy figures are targets to
> measure, not guarantees. A passing watermark check does not prove that
> all content is unchanged. A detector failure is unknown/unavailable
> evidence, not a clean result.

## 2. Problem Statement

### 2.1 Current risks to address

-   **Temporal edits:** A watermark may survive trimming, scene
    insertion, or reordering.
-   **Post-watermark manipulation:** A face swap or visual edit may be
    applied after watermarking.
-   **Benign transcoding:** Messaging and video platforms can change
    codecs, bitrate, metadata, and encoded bytes without materially
    changing visible content.
-   **Mutable records:** A local database can be altered by an actor
    with sufficient access. Hashes stored only in that database do not
    independently protect the history.
-   **Detector failure:** Failed frame extraction, missing model
    weights, or inference exceptions must not be interpreted as evidence
    that a file is clean.

### 2.2 Objective

Build an explainable, local, CPU-compatible verification pipeline that
combines provenance, watermark, temporal, metadata, forensic, and ledger
evidence. Preserve current public APIs and result schemas unless a
deliberate, reviewed change is necessary.

## 3. Design Principles

1.  Separate provenance linkage from content integrity.
2.  Treat perceptual similarity as approximate evidence, not
    cryptographic identity.
3.  Preserve explicit pass, concern, unavailable, and error states.
4.  Never let missing evidence become negative evidence.
5.  Keep the five existing final classifications.
6.  Calibrate thresholds using representative clean, transcoded, edited,
    and manipulated samples.
7.  Benchmark before claiming performance improvements.
8.  Record algorithm versions, configuration, and evidence provenance.
9.  Keep the system usable on Windows without a GPU.
10. Explain uncertainty and limitations in every final report.

## 4. Proposed Architecture

``` text
Input media
   |
   +--> Validate input and compute full-file SHA-256
   |
   +--> Asset lookup and watermark detection
   |
   +--> Media decode
   |      +--> Temporal perceptual fingerprints
   |      +--> Technical metadata extraction/comparison
   |      +--> Signal/frequency forensic analysis
   |      +--> Optional CPU-compatible ML inference
   |
   +--> Verify evidence-ledger integrity/checkpoints
   |
   v
Evidence consistency and central decision engine
   |
   v
Existing five classifications + explanation and limitations
```

Each subsystem should return structured evidence. Only the central
decision engine should assign the final classification.

## 5. Layer 1 --- Temporal Perceptual Fingerprinting

### Goal

Find evidence consistent with prefix/suffix trimming, internal frame
deletion, inserted segments, scene replacement, or scene reordering.

### Algorithm

1.  Sample frames deterministically, initially at a configurable
    interval such as one frame per second.
2.  Record each sample's timestamp and frame index where available.
3.  Convert each frame to grayscale and resize it to 9 × 8 pixels.
4.  Compare adjacent horizontal pixels to create a 64-bit difference
    hash (dHash).
5.  Store the ordered sequence of `(timestamp, hash)` values, along with
    sampling settings, algorithm version, decode status, and
    skipped-frame information.
6.  Compare the submitted sequence with the registered sequence using a
    transparent baseline matcher. Evaluate edit-distance or
    dynamic-time-warping alignment only after a baseline is tested.

For hashes (h_a) and (h_b), Hamming distance is:

\[ d_H(h_a,h_b)=`\operatorname{popcount}`{=tex}(h_a `\oplus `{=tex}h_b)
\]

Normalized distance is (d_N=d_H/64). A low distance indicates
fingerprint similarity, not pixel identity. A 60-sample sequence needs
480 bytes for the raw 64-bit hashes, excluding metadata and database
overhead.

### Output

Return availability/error status, match ratio, matched and unmatched
segments, possible trim/reordering, algorithm version, and an
explanation of limitations.

### Tests and acceptance

Test identical files, benign transcodes, prefix/suffix trims, internal
deletions, inserted and reordered segments, unrelated similar-looking
content, variable-frame-rate files, failed seeking, corrupt files, and
short clips. Results must be repeatable. Failed decoding must remain
unavailable/error, never an implicit clean result. Measure runtime and
memory.

Do not adopt a universal Hamming-distance threshold (for example, 12
bits) without calibration.

## 6. Layer 2 --- Technical Metadata Drift Analysis

### Goal

Compare technical properties and help distinguish likely transcoding
from material changes. Metadata differences alone cannot prove malicious
manipulation; metadata can be missing, rewritten, or forged.

### Extraction

Use `ffprobe` JSON output if available. Record tool version, exit
status, and timeout/error state.

-   **Container:** format, duration, bitrate, start time, relevant tags,
    encoder information.
-   **Video:** codec/profile/level, pixel format, dimensions, aspect
    ratios, frame rates, color properties, frame count where reliable,
    time base and stream duration.
-   **Audio:** codec, sample rate, channel count/layout, duration,
    bitrate where available.

### Comparison policy

-   **Technical match:** report fields that match; do not infer
    authenticity from metadata alone.
-   **Plausible transcode drift:** codec, bitrate, encoder tag, or audio
    encoding changes while other evidence remains consistent.
-   **Material differences requiring review:** substantial duration,
    aspect-ratio, dimension, frame-rate, or stream changes.

Proposed thresholds such as a 0.5-second duration change or 10--60%
bitrate reduction are hypotheses. Make them configurable and validate
across clip lengths, variable-frame-rate media, and real platform
transcodes.

### Tests

Test exact matches, codec and bitrate changes, dimension/aspect-ratio
changes, audio re-encoding, trims, missing metadata, malformed output,
missing `ffprobe`, timeouts, and multiple streams.

### Acceptance

Report exact differing fields. Tool failure must be distinct from a
successful comparison. Metadata alone must not establish an authentic or
manipulated verdict.

## 7. Layer 3 --- Watermark and Content Cross-Validation

### Critical design limitation

A SHA-256 hash of encoded stream bytes generally changes after lossy
transcoding. Therefore, distinguish:

-   full-file SHA-256 for exact file identity;
-   full stream hash for exact encoded-stream identity;
-   perceptual fingerprints for approximate content comparison.

Do not require a raw-stream hash match for all legitimate derivatives. A
short hash fragment has a smaller collision margin than a full hash and
must not be the sole security check.

### Proposed approach

1.  Keep full-file SHA-256 for exact identity.
2.  Treat a watermark as evidence of a possible relationship to a
    registered asset, subject to confidence gating and registered-token
    resolution.
3.  Optionally record a full stream hash for exact stream comparisons.
4.  For transcoded media, compare temporal fingerprints and metadata
    separately.
5.  Version any changed watermark payload and preserve
    backward-compatible decoding where feasible.
6.  Document exactly which bytes or canonical representation a content
    hash covers and when it is computed.
7.  Never let a watermark match suppress contradictory forensic
    evidence.

### Tests and acceptance

Test registered and unknown IDs, low-confidence registered IDs,
compression/resizing, re-encoding, post-watermark edits, payload
corruption, legacy payloads, detector failures, and stream-hash mismatch
after transcode. An ID alone must never bypass confidence gating.
Transcoding must not be rejected solely because encoded bytes changed.

## 8. Layer 4 --- Lightweight CPU Forensic Analysis

### Tier 0: Frequency-domain features

Evaluate DCT/FFT residual features on selected frames or face crops:

1.  Extract frames and detect/crop faces when supported.
2.  Normalize crops consistently.
3.  Compute documented frequency-domain features.
4.  Aggregate across frames while retaining per-frame evidence.
5.  Return uncertainty, availability, and extraction status.

Frequency artifacts are not unique to synthetic media; compression,
resizing, sharpening, camera pipelines, and scene content can affect
them. Validate before using the signal in final decisions.

### Tier 1: Small ONNX model

Evaluate MesoInception-4 or another suitably licensed, reproducible
model. Verify model provenance/license, preprocessing, actual quantized
file size, ONNX Runtime CPU support, output-label semantics,
calibration, and false-positive/false-negative rates.

### Tier 2: Conditional model

Consider a second-stage model only if it is actually trained and
validated for deepfake detection. A generic image classifier is not
automatically a deepfake detector. A proposed uncertainty band such as
0.40--0.70 is an experiment, not a validated threshold.

### Integration requirements

Inspect existing `forensics/video_ml.py`, `forensics/audio_ml.py`, and
`forensics/service.py` before adding adapters. Failed frame extraction,
missing weights, inference exceptions, malformed outputs, and non-finite
confidence must be explicit unavailable/error states---not clean
predictions.

### Tests and acceptance

Test clean/manipulated samples, compression, resizing, no-face and
multi-face media, failed extraction, missing weights, exceptions,
malformed outputs, non-finite confidence, and CPU-only inference. Record
model/preprocessing versions and measure quality and resource use. No
uncalibrated score may independently produce an "authentic" verdict.

## 9. Layer 5 --- Tamper-Evident Evidence Ledger

### Goal

Make unauthorized changes to registration and verification events
detectable while keeping operation local and offline-capable.

A hash chain is only as trustworthy as its signing keys and trusted
checkpoints. An attacker able to rewrite the database and replace all
unprotected keys/checkpoints may be able to rewrite history.

### Event format

Define a versioned event containing sequence index, timestamp, previous
block hash, asset ID, operation, canonical payload or payload hash,
signer key ID, signature, and current block hash. Specify deterministic
serialization, field order, encoding, timestamps, and optional-field
behavior.

Conceptually:

\[
H_n=`\operatorname{SHA256}`{=tex}(`\operatorname{CanonicalEncode}`{=tex}(B_n))
\]

Each event links to the preceding event. Independent verification must
reproduce the same hashes.

### Ed25519 signing

If an approved maintained library is available, define key generation,
private-key protection, public-key identification, signing,
verification, rotation/revocation, backup, and missing/invalid-key
behavior. Do not store an unprotected private key alongside the mutable
database and then claim independent trust.

### Merkle proofs and checkpoints

Define leaf ordering, odd-node behavior, serialization, root generation,
and inclusion proof verification. Export signed checkpoints to a
separately protected location. A root stored only in the same mutable
database is not an independent trust anchor.

### Tests and acceptance

Test valid chains, modified payloads, reordered/deleted events, altered
previous hashes, invalid signatures, unknown keys, key rotation,
malformed serialization, Merkle proofs, rollback/truncation, and missing
checkpoints. Integrity failure must be visible and must not be silently
ignored. Cryptographic integrity alone does not guarantee legal
admissibility.

## 10. Evidence Aggregation and Final Classification

### Evidence contract

Each subsystem should report its name/version, status, measured values,
threshold/configuration, asset reference, explanation, timestamp, and
limitations.

### Intended classification semantics

  -----------------------------------------------------------------------
  Classification                      Intended meaning
  ----------------------------------- -----------------------------------
  **Verified Original**               Strong evidence that the file is
                                      the exact registered original,
                                      required integrity checks pass, and
                                      no unresolved contradiction
                                      remains.

  **Verified Derivative**             Strong evidence linking the file to
                                      a registered asset, with expected
                                      transformations accounted for and
                                      no unresolved evidence invalidating
                                      the claim.

  **Potentially Manipulated**         Sufficiently supported evidence
                                      indicates likely manipulation,
                                      unexplained content changes, or a
                                      material integrity conflict.

  **Unknown Provenance**              The system cannot establish a
                                      reliable relationship to a
                                      registered source.

  **Insufficient Evidence**           Required evidence is missing,
                                      unavailable, failed, or too
                                      ambiguous for a defensible
                                      decision.
  -----------------------------------------------------------------------

Confirm these semantics against the existing decision logic before
changing it. A legitimate derivative can be edited; the report should
describe what kind of derivative is being assessed.

### Conflict rules

-   A registered watermark plus strong manipulation evidence must not
    automatically yield Verified Derivative.
-   A missing model must not count as a clean result.
-   A hash mismatch may result from legitimate transcoding and is not
    automatically malicious.
-   A failed ledger check is a custody-integrity concern.
-   Model confidence must not override input validation or detector
    errors.
-   Missing required evidence should preserve uncertainty.

Avoid arbitrary composite-score cutoffs (such as 90/100) unless the
score is calibrated and its meaning is documented. Always retain the
underlying evidence and decision rationale.

## 11. End-to-End Workflows

### Registration

1.  Validate media readability and expected decoding.
2.  Compute full-file SHA-256.
3.  Extract metadata and record tool status/version.
4.  Sample frames deterministically and compute temporal fingerprints.
5.  Create the asset record using the existing registration flow.
6.  Embed a watermark if requested.
7.  Hash the final output and record its relationship to the source.
8.  Append registration/watermarking ledger events.
9.  Sign events if signing is enabled and keys are available.
10. Read back and verify the new record and ledger.

If watermarking changes the file, record original and output hashes
separately.

### Verification

1.  Validate input and compute full-file hash.
2.  Check exact registered-file identity.
3.  Run watermark detection with confidence gating and registered-asset
    resolution.
4.  Extract and compare metadata.
5.  Compute and align temporal fingerprints when a registered sequence
    exists.
6.  Run configured forensic signals and optional ML models.
7.  Verify ledger and trusted checkpoints where configured.
8.  Aggregate evidence without converting unavailable/error states into
    negative findings.
9.  Apply the existing classification rules.
10. Produce a report containing the classification, evidence,
    explanation, limitations, and detector availability.

## 12. Hardware and Performance Targets

Reference target: entry-level Intel CPU, 4 GB RAM, integrated graphics,
no GPU. All figures are targets to measure, not established guarantees.

  -----------------------------------------------------------------------
  Component               Target approach         Validation
  ----------------------- ----------------------- -----------------------
  Temporal dHash          Low-memory,             Benchmark duration,
                          deterministic           resolution, codecs
                          extraction              

  Metadata extraction     Small overhead relative Measure startup and
                          to decoding             processing time

  Watermark checks        Stay within local CPU   Benchmark engines and
                          budget                  catalog sizes

  Frequency features      No model weights        Measure per-frame cost
                          required                and false positives

  Small ONNX model        Prefer small validated  Measure actual file
                          artifact                size, peak RAM, latency

  Second-stage model      Run only when justified Compare cascade
                                                  quality/cost with
                                                  always-on inference

  Ledger                  Low overhead relative   Benchmark append,
                          to media analysis       signature checks,
                                                  full-chain verification
  -----------------------------------------------------------------------

### Benchmark methodology

Use identical inputs/configuration; apply consistent warm-ups and
repeated runs; report median and spread; record OS, CPU, Python, NumPy,
OpenCV, FFmpeg, and ONNX Runtime versions; measure peak memory; and
report failures/skipped inputs. Benchmark clean, manipulated,
compressed, short, and long media separately. Do not claim a total
verification time until the integrated pipeline is measured on target
hardware.

## 13. Proposed Project Structure

This is a proposal, not an instruction to create every file at once.
Reuse current modules and conventions.

``` text
MediaProof/
├── core/
│   └── security/
│       ├── __init__.py
│       ├── chain_ledger.py
│       ├── signer.py
│       └── merkle.py
├── forensics/
│   ├── config.py
│   ├── temporal_dhash.py
│   ├── metadata_drift.py
│   ├── ml_frequency.py
│   ├── video_ml.py
│   └── service.py
└── tests/
    ├── test_temporal_dhash.py
    ├── test_metadata_drift.py
    ├── test_chain_ledger.py
    ├── test_signer.py
    ├── test_merkle.py
    └── integration/
        └── test_end_to_end_verification.py
```

Inspect the existing layout and result/configuration abstractions before
creating modules.

## 14. Phased Implementation Roadmap

### Phase 0 --- Baseline and safety audit

**Tasks** - Run the current test suite and record exact results. -
Record classifications for controlled fixtures. - Trace current decision
logic and unavailable/error handling. - Measure baseline runtime and
memory. - Document APIs/result schemas and known high-impact defects.

**Deliverables:** baseline test report, decision-flow diagram, risk
register, representative media manifest.

**Exit criteria:** current behavior is documented and confirmed
correctness defects are prioritized.

### Phase 1 --- Temporal fingerprints and metadata drift

**Tasks** 1. Implement deterministic dHash extraction. 2. Store
fingerprints compatibly with current asset records. 3. Implement a
baseline sequence comparator. 4. Add metadata extraction with explicit
failure states. 5. Add field-by-field drift reporting. 6. Add
unit/integration tests and benchmarks.

**Exit criteria:** controlled edits produce documented results; benign
transcodes are not automatically classified as malicious; failures
remain unavailable/error; API changes are avoided or documented.

### Phase 2 --- Tamper-evident ledger

**Tasks** 1. Define canonical event serialization. 2. Implement
hash-chained events. 3. Implement Ed25519 signing/verification. 4.
Define key storage and rotation. 5. Implement Merkle roots and inclusion
proofs. 6. Add separately preserved checkpoints. 7. Test modification,
deletion, reordering, and key failures.

**Exit criteria:** independent verification detects altered events,
key/checkpoint limitations are documented, and failures are visible.

### Phase 3 --- Lightweight forensic cascade

**Tasks** 1. Validate current detector failure semantics. 2. Add
frequency features as an experimental signal. 3. Select a licensed and
reproducible ONNX model only after artifact evaluation. 4. Validate
preprocessing/output semantics. 5. Add conditional inference only if it
improves measured quality/cost. 6. Evaluate across clean, manipulated,
compressed, and out-of-domain media.

**Exit criteria:** no failure is reported as clean, model metrics and
limitations are documented, and resource use is measured.

### Phase 4 --- Evidence integration

**Tasks** 1. Define each subsystem's evidence-state contract. 2. Map
hash, watermark, temporal, metadata, forensic, and ledger evidence into
the central decision engine. 3. Specify conflict handling and
missing-evidence rules. 4. Preserve the five classifications. 5. Add
explanations and conflicting-signal integration tests.

**Exit criteria:** contradictory evidence is not silently overridden;
missing detectors cannot create a clean result; classifications follow
explainable rules.

### Phase 5 --- Evaluation and release readiness

**Tasks** 1. Assemble a versioned evaluation corpus. 2. Measure false
positives/negatives per signal and for combined decisions. 3. Benchmark
CPU time and peak memory on target hardware. 4. Test malformed, corrupt,
and unusually large inputs. 5. Verify Windows setup and offline
behavior. 6. Document model licenses and optional assets. 7. Test
portable packaging separately from development startup. 8. Publish
limitations and reproducible test instructions.

**Exit criteria:** results are documented, unsupported
accuracy/performance guarantees are removed, and high-severity issues
are resolved before release.

## 15. Evaluation Dataset and Metrics

Include exact originals, watermarked outputs, benign transcodes,
resized/cropped files, trims, deletions, inserted/reordered scenes,
visual edits/face swaps, unrelated media, corrupt inputs, and
missing-tool/model cases.

For every sample, record ground truth, source, transformation, tool
version, and expected limitations.

Measure: - false-positive rate on clean and benignly transcoded media; -
false-negative rate by manipulation type; - precision/recall where
labels are meaningful; - detector availability/failure rate; -
classification confusion matrix; - runtime and peak memory; - results by
codec, resolution, duration, and compression.

Do not claim broad deepfake-detection accuracy unless the corpus and
evaluation procedure support the claim.

## 16. Risks and Mitigations

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  Perceptual hashes miss subtle edits Use as one signal and document
                                      known failure cases

  Benign transcodes trigger warnings  Calibrate on real transcodes and
                                      report field-level differences

  Watermark survives manipulation     Require independent checks and
                                      preserve contradictory evidence

  Short hashes collide                Use full hashes for cryptographic
                                      identity

  ML models are biased or out of      Evaluate across sources, codecs,
  domain                              and manipulation types

  Detector failure looks clean        Explicit unavailable/error states
                                      and integration tests

  Database and keys are rewritten     Protect keys and preserve signed
  together                            checkpoints separately

  Composite score hides               Explicit rules and retained
  contradictions                      underlying evidence

  Performance targets are unrealistic Benchmark on target hardware

  Scope expands too quickly           Phase gates, small changes, and
                                      independent reviews
  -----------------------------------------------------------------------

## 17. Immediate Next Steps

1.  Complete a baseline and safety audit of current behavior.
2.  Fix confirmed correctness and fail-safe defects before adding new
    features.
3.  Prototype temporal fingerprints as an isolated module.
4.  Add metadata drift as a separate evidence source.
5.  Design the ledger threat model before implementing cryptography.
6.  Evaluate candidate lightweight models before adopting them.
7.  Integrate new signals only after isolated tests and measured
    behavior are satisfactory.

### Recommended first milestone

Begin with **Phase 0 --- Baseline and safety audit**. Then fix the
smallest confirmed correctness issue with regression tests. Once
baseline behavior is stable, proceed to **Phase 1 --- Temporal
fingerprints and metadata drift**.

This order avoids layering new signals onto decision logic whose failure
semantics have not yet been validated.
