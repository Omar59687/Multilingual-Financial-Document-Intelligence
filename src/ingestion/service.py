"""Ingestion service: ingest_document() + ingest_directory().

Domain failures (missing/unsupported/corrupt files, parser exceptions)
become controlled FAILED/UNSUPPORTED results — never raised. Only
programmer errors (e.g. output dir not writable) raise.
"""

import hashlib
import time
from pathlib import Path

from .model import (Document, FileType, IngestedDocument, IngestionStatus,
                    LanguageHint, document_id_for)
from .parsers import (parse_csv, parse_docx, parse_image, parse_pdf,
                      parse_xlsx)
from .router import SUPPORTED_SUFFIXES, route

PARSERS = {
    FileType.PDF: parse_pdf,
    FileType.DOCX: parse_docx,
    FileType.XLSX: parse_xlsx,
    FileType.CSV: parse_csv,
    FileType.PNG: lambda p, i: parse_image(p, i, FileType.PNG),
    FileType.JPG: lambda p, i: parse_image(p, i, FileType.JPG),
}

# Never ingested as corpus documents (evaluation sidecars live beside docs).
SKIP_NAMES = {"manifest.json"}
SKIP_DIRS = {"ground_truth", "__pycache__"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failed(document_id: str, filename: str, status: IngestionStatus,
            error: str, latency_ms: float) -> IngestedDocument:
    doc = Document(document_id=document_id, filename=filename,
                   file_type=None, language_hint=LanguageHint.UNKNOWN,
                   status=status, error=error, latency_ms=latency_ms)
    return IngestedDocument(document=doc, ok=False, error=error,
                            latency_ms=latency_ms)


def ingest_document(path) -> IngestedDocument:
    """Ingest one file. Deterministic IDs; single parse per call."""
    started = time.perf_counter()
    path = Path(path)

    def elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000.0

    routed = route(path)
    if routed.status is not None:  # missing / unreadable / unsupported
        doc_id = path.name if routed.status == IngestionStatus.FAILED \
            else f"UNSUPPORTED-{path.name}"
        return _failed(doc_id, path.name, routed.status, routed.reason,
                       elapsed_ms())
    try:
        content_hash = _sha256(path)
    except OSError as exc:
        return _failed(path.name, path.name, IngestionStatus.FAILED,
                       f"unreadable: {exc}", elapsed_ms())
    doc_id = document_id_for(path, content_hash)
    try:
        document = PARSERS[routed.file_type](path, doc_id)
    except Exception as exc:  # parser exception -> controlled FAILED
        return _failed(doc_id, path.name, IngestionStatus.FAILED,
                       f"{routed.file_type.value} parser failed: "
                       f"{type(exc).__name__}: {exc}", elapsed_ms())
    document.latency_ms = elapsed_ms()
    latency = document.latency_ms
    ok = document.status not in (IngestionStatus.FAILED,
                                 IngestionStatus.UNSUPPORTED)
    return IngestedDocument(document=document, ok=ok,
                            error=document.error, latency_ms=latency)


def ingest_directory(path, include_suffixes=None) -> dict:
    """Ingest a directory: sorted discovery, per-file isolation, summary.

    Skips ground-truth sidecars (`*.json`, `ground_truth/` dirs) — they are
    evaluation fixtures, never corpus. Returns {"results": [...], "summary"}.
    """
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")
    allowed = set(include_suffixes) if include_suffixes \
        else SUPPORTED_SUFFIXES
    skipped, all_files = [], sorted(
        p for p in path.rglob("*")
        if p.is_file() and not p.name.startswith("."))
    candidates = []
    for candidate in all_files:
        if candidate.name in SKIP_NAMES or \
                any(part in SKIP_DIRS for part in candidate.parts):
            skipped.append(candidate.name)  # evaluation fixture, not corpus
        elif candidate.suffix.lower() in allowed:
            candidates.append(candidate)
    results, failed = [], []
    for candidate in candidates:
        if candidate.suffix.lower() == ".json":
            skipped.append(candidate.name)  # ground-truth sidecar: not corpus
            continue
        result = ingest_document(candidate)
        results.append(result)
        if not result.ok:
            failed.append(candidate.name)
    by_status: dict = {}
    for result in results:
        key = result.document.status.value
        by_status[key] = by_status.get(key, 0) + 1
    return {
        "results": results,
        "summary": {
            "directory": str(path),
            "files_found": len(all_files),
            "ingested": len(results),
            "skipped_sidecars": skipped,
            "failed": failed,
            "by_status": by_status,
            "total_latency_ms": sum(r.latency_ms for r in results),
        },
    }
