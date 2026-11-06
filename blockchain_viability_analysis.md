# Blockchain Viability Analysis for MediaProof
## Based on IEEE Paper: *SecEvidence — Litigation Evidence Preservation and Verification on Blockchain*

---

## 1. What the Paper Proposes (Summary)

**SecEvidence** is a consortium blockchain framework built by Shanghai Pudong Thunisoft (a legal tech firm) for Chinese courts. Its architecture has three tiers:

| Layer | What It Does |
|---|---|
| **Basic Platform** | Tendermint consensus engine, distributed ledger, smart contracts |
| **Business Middleware** | Real-name authentication, data storage verification, signature + seal mutual recognition |
| **Upper Layer Apps** | Litigation online cert, process cert, evidence authenticity cert |

**Core technical components:**
- **SM2P256V1** elliptic curve digital signatures (Chinese state crypto standard — alternative to Ed25519/ECDSA)
- **SM3** hash algorithm (alternative to SHA-256, similar computational structure)
- **SHA-512** as an alternative high-security hash
- **BFT consensus** requiring minimum 4 nodes (tolerates 1 Byzantine faulty node)
- **Tendermint v0.33** as the blockchain core (Golang)

**What they proved experimentally (2GB memory, Intel Xeon @ 2.5GHz):**
- SM2 signing: ~1.25ms avg; SHA-256: ~0.001ms; SM3: ~0.006ms
- Switching to stronger algorithms adds **<2ms overhead total** — negligible
- Block generation delay is dominated by **network transmission**, not crypto

---

## 2. Viability Assessment: Point-by-Point Against MediaProof

### ✅ VIABLE — Concept A: Cryptographic Hash Anchoring on a Local Chain

**What SecEvidence does:**  
Before storing evidence, they hash the data (SHA-256 / SM3) and publish that hash to an immutable blockchain ledger. The blockchain ensures *no authority can unilaterally tamper* with the record.

**What MediaProof does today:**  
SHA-256 hash stored in SQLite. SQLite is mutable — anyone with filesystem access can alter or delete records. A file can be re-registered with a new hash after tampering.

**What we can borrow:**  
A **local append-only hash chain** — a lightweight blockchain-inspired log where each entry chains to the previous via SHA-256, making retroactive tampering detectable **without needing any network or consensus nodes**.

```
Block 0 (genesis):
  asset_id: MP-xxxx
  sha256: abc...
  fingerprint_hash: def...
  timestamp: 2026-10-15T10:00:00Z
  prev_hash: 0000000000000000

Block 1:
  asset_id: MP-yyyy
  sha256: 789...
  fingerprint_hash: ...
  timestamp: 2026-10-15T10:05:00Z
  prev_hash: SHA256(Block 0)    ← breaks if Block 0 is altered
```

This is **100% local, CPU-only, zero network, zero consensus nodes** — just SHA-256 chaining of SQLite rows. Tamper detection: if any past block's `prev_hash` doesn't match the next block's actual hash, the chain is broken.

**Feasibility for MediaProof:** ✅ Very HIGH. Pure Python + SQLite. ~5ms overhead per registration.

---

### ✅ VIABLE — Concept B: Cryptographic Timestamp Commitment

**What SecEvidence does:**  
Each piece of evidence gets a blockchain timestamp when it's submitted. This timestamp is immutable once on-chain — proving the file existed at that moment.

**What MediaProof does today:**  
`created_at` and `registered_at` are stored as plain ISO strings in SQLite. Trivially editable.

**What we can borrow:**  
At registration, compute a **commitment hash** = SHA-256(asset_id + sha256 + registered_at + watermark_token) and store it in the chain log. At verification, re-derive the commitment and check chain integrity.

This creates a **tamper-evident timestamp record**: if the database is modified post-registration, the commitment will no longer match.

**Feasibility for MediaProof:** ✅ Very HIGH. 2 lines of code added to `ProvenanceService.register_media()`.

---

### ✅ VIABLE — Concept C: ECDSA Registration Signature

**What SecEvidence does:**  
Every evidence submission is digitally signed by the submitter's private key (SM2 or Ed25519). Validators verify with the public key.

**What we can borrow:**  
When a user registers a video in MediaProof, generate a **device keypair** (Ed25519) once and sign the registration record:
```
signature = Ed25519.sign(private_key, SHA256(asset_id + sha256 + timestamp))
```
Store the `signature` and `public_key` in the database. At verification, re-derive the message and verify the signature.

**Why this matters for MediaProof:**  
Even if the SQLite database is copied and modified on another machine, the signature is tied to the original device's private key. Modified records will fail signature verification.

**Feasibility for MediaProof:** ✅ HIGH. Python's `cryptography` library (already likely present). ~1–2ms overhead.

---

### ⚠️ PARTIALLY VIABLE — Concept D: Multi-Party Verification

**What SecEvidence does:**  
Multiple institutions form a consortium chain. A judge cannot unilaterally alter evidence because all court nodes + third-party certification nodes hold the same ledger.

**What MediaProof is:**  
A **single-machine, local-only, portable tool**. There are no other nodes. There is no network.

**What is transferable (adapted):**  
- Export a **signed provenance manifest** (JSON + ECDSA signature) so a second MediaProof instance on another machine can import and verify it independently
- The manifest acts as a "portable chain record" that doesn't require consensus — just signature verification against the original public key

**Limitation:**  
Without a second party or a network, "multi-party" reduces to "device-bound signing" — better than nothing but not true decentralization.

**Feasibility for MediaProof:** ⚠️ MEDIUM. Architecturally clean, but the trust model is limited to "trust the originating device."

---

### ❌ NOT VIABLE — Full Blockchain Network (Tendermint / Distributed Ledger)

**What SecEvidence uses:**  
Tendermint v0.33 (Golang), requiring:
- Minimum 4 nodes for BFT consensus
- Network infrastructure between nodes
- Golang + GMSSL / OpenSSL compiled libraries
- 2GB+ RAM for node operation

**Why it does NOT fit MediaProof:**
1. MediaProof is a **portable, local, single-binary tool** — no external services, no network, no dependencies
2. Running a blockchain node requires persistent background services + network ports
3. BFT consensus requires at least 4 machines — impossible on a single device
4. Golang runtime cannot be bundled inside a Python PyInstaller portable exe
5. SM2/SM3/SM4 (Chinese state crypto) are overkill and have limited Python library support

**Verdict:** ❌ Cannot implement full Tendermint blockchain in MediaProof without fundamentally changing its architecture to a server-based or networked product.

---

### ❌ NOT VIABLE — Smart Contracts

**What SecEvidence uses:**  
Smart contracts automate arbitration logic and evidence lifecycle rules on the blockchain (e.g., when a case is filed, the contract triggers evidence locking).

**Why not viable:**  
Smart contracts require a blockchain runtime (EVM, Tendermint ABCI, etc.) — same infrastructure issue as above. Entirely out of scope for a local portable tool.

---

## 3. What the Paper Specifically Validates That Helps Us

Even though full blockchain is not viable, the paper's **experimental data** gives us useful confirmation:

| Paper Finding | Implication for MediaProof |
|---|---|
| SHA-256 on 2GB RAM: **0.001ms average** | Our SHA-256 file hashing is extremely cheap even on budget hardware |
| SM3 (stronger hash): **0.006ms** — 6× slower but still sub-millisecond | If we add chaining hashes (commitment records), it's negligible overhead |
| Ed25519 signing: **0.055ms** | Registration signatures are essentially free performance-wise |
| SM2 signing: **1.25ms** | Python `cryptography` Ed25519 is fast; even SM2-equivalent overhead is <2ms |
| Network latency > all crypto cost | Our bottleneck is video processing (FFmpeg), not any cryptographic operation |

---

## 4. Recommended Implementation: Local Append-Only Chain Log

This is the **correct, viable, MediaProof-native adaptation** of the blockchain concept:

### Architecture

```
provenance/chain.py  (NEW)
  ┌─────────────────────────────────────────────────────┐
  │  ChainBlock                                          │
  │    block_index:    INTEGER (auto-increment)          │
  │    asset_id:       TEXT                              │
  │    event_type:     TEXT  (REGISTER / VERIFY / ...)  │
  │    payload_hash:   TEXT  (SHA-256 of event data)    │
  │    prev_block_hash:TEXT  (SHA-256 of previous block)│
  │    timestamp:      TEXT  (ISO UTC)                   │
  │    signature:      TEXT  (Ed25519 signature, b64)    │
  │    public_key:     TEXT  (Ed25519 pub key, b64)      │
  └─────────────────────────────────────────────────────┘
```

New SQLite table: `provenance_chain`

### What it enables

| Capability | How |
|---|---|
| **Tamper detection on DB** | Chain verification: re-hash each block's data and check prev_hash links |
| **Proof original video existed at time T** | Block timestamp + payload_hash + Ed25519 signature |
| **Proof no record was deleted** | Gap detection: block_index must be strictly sequential, no gaps |
| **Proof of verification events** | Each `/api/verify` call adds a VERIFY block with result hash |
| **Portable evidence export** | Export chain as signed JSON manifest; import on another machine |

### Chain Integrity Verification (new API endpoint)

`GET /api/chain/verify` — walks the entire chain:
1. For each block: recompute `payload_hash` from stored fields
2. Verify `prev_block_hash` matches SHA-256 of previous block
3. Verify Ed25519 signature on each block
4. Report: `CHAIN_INTACT` or `CHAIN_BROKEN` at specific block index

---

## 5. Final Verdict

| Concept | From SecEvidence | Viability for MediaProof | Effort |
|---|---|---|---|
| Local append-only hash chain | ✅ Core concept | ✅ HIGH | Medium |
| Cryptographic timestamp commitment | ✅ On-chain timestamp | ✅ HIGH | Low |
| Ed25519 registration signature | ✅ ECDSA signing | ✅ HIGH | Low |
| Signed portable manifest export | ✅ Consortium cross-chain | ⚠️ MEDIUM | Medium |
| Full Tendermint blockchain | Core infrastructure | ❌ NOT VIABLE | N/A |
| BFT multi-node consensus | Core infrastructure | ❌ NOT VIABLE | N/A |
| Smart contracts | Upper layer | ❌ NOT VIABLE | N/A |
| SM2/SM3 algorithms | Algorithm layer | ⚠️ OPTIONAL | Low (if needed) |

### Bottom Line

> **Yes, blockchain concepts ARE viable for MediaProof — but not full blockchain.**  
> The paper's contribution to us is its *design philosophy*: hash chaining + cryptographic signing of evidence records makes tampering with the provenance database **detectable** and **attributable**.  
>  
> A local append-only chain (implemented as a second SQLite table + Ed25519 signing) gives MediaProof the same tamper-evidence guarantee that SecEvidence gets from Tendermint — without any network, nodes, or infrastructure overhead.  
>  
> This directly addresses the **single biggest architectural weakness** in MediaProof today: the SQLite database is fully mutable, so an attacker who gains filesystem access can silently modify or delete provenance records.

---

## 6. Where This Fits in the Overall Enhancement Plan

This becomes **Phase 4: Chain Integrity Layer** — added after Pillars A, B, C:

```
Phase 1: Perceptual Fingerprint (dHash temporal sequence)
Phase 2: Metadata Drift Comparison
Phase 3: Watermark Hash Fragment Validation
Phase 4: Local Append-Only Provenance Chain (NEW — from this paper)
  ├── provenance/chain.py            — block model + chain logic
  ├── provenance/chain_repository.py — SQLite CRUD for chain
  ├── provenance/chain_service.py    — register/verify chain events
  └── backend/main.py                — GET /api/chain/verify endpoint
```
