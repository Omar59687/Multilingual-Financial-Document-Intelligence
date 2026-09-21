"""MizanIQ OCR/vision benchmark tooling (Phase 2, evaluation-side only).

Scoring helpers, benchmark-truth derivation, and the experiment result
contract. NEVER imported by production code (src/ingestion); guarded by
test. No OCR/AI/model dependencies here — stdlib only.
"""
