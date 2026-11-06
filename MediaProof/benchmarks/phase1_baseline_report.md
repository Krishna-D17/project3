# MediaProof Phase 1 -- Baseline Performance Benchmarks

> [!IMPORTANT]
> **Measurement Notice & Memory Scope:**
> 1. Measurements represent results obtained in the tested local execution environment and are empirical baselines for regression detection, not universal guarantees.
> 2. **Memory Scope:** Peak memory measures strictly Python-tracked heap memory via Python's standard `tracemalloc` module. It **does NOT** measure total process memory (Resident Set Size / Working Set) or allocations performed in native C/C++ extensions (e.g. OpenCV image buffers) or spawned subprocesses (e.g. FFmpeg / FFprobe). Total process memory was not measured.

## 1. Execution Environment

| Attribute | Value |
| :--- | :--- |
| **OS** | Windows 10 (build 10.0.26300) |
| **Architecture** | AMD64 |
| **Processor** | Intel64 Family 6 Model 154 Stepping 4, GenuineIntel |
| **Logical CPU Cores** | 12 |
| **Python Version** | 3.11.14 (CPython) |
| **OpenCV Version** | 5.0.0 |
| **FFmpeg Available** | Yes |
| **FFmpeg Version** | ffprobe version 7.0.2-essentials_build-www.gyan.dev Copyright (c) 2007-2024 the FFmpeg developers |
| **FFmpeg Path** | `C:\Users\krish\Desktop\project3\MediaProof\tools\ffmpeg\ffmpeg.exe` |
| **FFprobe Path** | `C:\Users\krish\Desktop\project3\MediaProof\tools\ffmpeg\ffprobe.exe` |

## 2. Benchmark Results

| Workload | Component | Input Spec | Mean Wall Time | Min / Max | Peak Python Heap (tracemalloc) | Status | Contract Validation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dhash_extract_short` | Temporal dHash Extractor | Short 2.0s clip (160x120 @ 10fps, sampling 0.5s) | 24.21 ms (±0.51) | 23.57 / 24.81 ms | 115.4 KB | SUCCESS | Validated: status=SUCCESS |
| `dhash_extract_medium` | Temporal dHash Extractor | Medium 8.0s clip (320x240 @ 15fps, sampling 0.5s) | 108.65 ms (±8.18) | 104.48 / 123.25 ms | 458.3 KB | SUCCESS | Validated: status=SUCCESS |
| `dhash_compare_short` | Temporal Sequence Comparer | 4-sample vs 4-sample comparison (semi-global DP) | 0.48 ms (±0.05) | 0.45 / 0.55 ms | 6.1 KB | SUCCESS | Validated: status=SUCCESS |
| `dhash_compare_medium` | Temporal Sequence Comparer | 16-sample vs 16-sample comparison (semi-global DP) | 3.28 ms (±1.19) | 2.41 / 5.33 ms | 26.0 KB | SUCCESS | Validated: status=SUCCESS |
| `metadata_extract_short` | Technical Metadata Extractor | FFprobe container & stream inspection (short clip) | 60.94 ms (±0.76) | 59.81 / 61.84 ms | 49.7 KB | SUCCESS | Validated: status=SUCCESS |
| `metadata_extract_medium` | Technical Metadata Extractor | FFprobe container & stream inspection (medium clip) | 61.93 ms (±0.65) | 61.03 / 62.83 ms | 49.0 KB | SUCCESS | Validated: status=SUCCESS |
| `metadata_compare_short_transcoded` | Technical Metadata Comparer | Strict partition difference evaluation (short vs transcoded) | 1.11 ms (±0.06) | 1.05 / 1.17 ms | 29.6 KB | SUCCESS | Validated: status=SUCCESS_WITH_DRIFT |
| `metadata_compare_medium_identical` | Technical Metadata Comparer | Strict partition difference evaluation (medium vs self) | 0.97 ms (±0.02) | 0.96 / 1.01 ms | 29.2 KB | SUCCESS | Validated: status=SUCCESS_WITH_MISSING_FIELDS |

## 3. Analysis and Architectural Observations

- **Temporal dHash Extraction:** Dominated by OpenCV frame decoding and scaling. Extraction is linear with sample count and proportional to frame resolution.
- **Temporal Sequence Alignment:** O(M*N) Dynamic Programming over 64-bit integer Hamming comparisons is extremely fast (< 1 ms for standard video clip sample sequences).
- **Technical Metadata Extraction:** Dominated by FFprobe process spawn and JSON demuxing overhead (~50-70 ms per invocation on Windows).
- **Technical Metadata Comparison:** Pure in-memory dictionary and stream affinity evaluation executes in sub-millisecond time (< 1 ms).
- **Memory Scope Clarification:** Peak heap values reflect allocations tracked by Python's `tracemalloc` allocator (< 500 KB peak Python heap). Total process memory (RSS) includes native C++ runtimes and was not measured.