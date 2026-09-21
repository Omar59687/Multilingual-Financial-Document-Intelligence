"""PaddleOCR-VL-1.6 benchmark adapter (evaluation-side, stdlib only).

Covers the Kaggle-side integration contract WITHOUT importing Paddle here:

- identity constants (candidate / model id / pipeline version — locked),
- PaddleSession: lazy, once-per-session pipeline creation with init
  latency measurement (importer + clock injectable for tests),
- adapt_output: defensive normalization of structured pipeline outputs
  into (text, fields, tables, visual_description, notes),
- blocker_result / build_result: schema-conformant result construction.

No Paddle import, no weights, no inference in this module — the real
``from paddleocr import PaddleOCRVL`` call happens only inside the Kaggle
notebook (mirrored logic) via an injected importer. Never hand-corrects
model text/numbers; unknown shapes are preserved + noted, never dropped.
"""

import time

from . import schema

CANDIDATE = "paddleocr-vl"
MODEL_ID = "PaddleOCR-VL-1.6"
PIPELINE_VERSION = "v1.6"

# First-baseline pipeline posture: official v1.6 defaults, NO overrides.
# Recorded (not tuned): orientation classification, unwarping, layout
# detection, chart recognition all stay at whatever the official pipeline
# does by default. Any future non-default option must be added here and
# recorded in the exported identity.
PIPELINE_OPTIONS = {
    "doc_orientation_classify": "official-default (not overridden)",
    "doc_unwarping": "official-default (not overridden)",
    "layout_detection": "official-default (not overridden)",
    "chart_recognition": "official-default (not overridden)",
}


def identity_for(paddleocr_version=None, paddle_version=None, device=None):
    """Reproducibility identity stamped on every Paddle result/meta."""
    return {
        "candidate": CANDIDATE,
        "model_id": MODEL_ID,
        "pipeline_version": PIPELINE_VERSION,
        "paddleocr_version": paddleocr_version or "unknown",
        "paddle_version": paddle_version or "unknown",
        "device": device or "unknown",
        "pipeline_options": dict(PIPELINE_OPTIONS),
    }


class PaddleSession:
    """Lazily creates the v1.6 pipeline exactly once per session.

    importer: zero-arg callable returning the PaddleOCRVL CLASS
      (default performs the real ``from paddleocr import PaddleOCRVL``).
      Tests inject a fake; the Kaggle notebook uses the default.
    clock: time.perf_counter-compatible (injectable for tests).
    """

    def __init__(self, importer=None, clock=None):
        self._importer = importer or self._default_importer
        self._clock = clock or time.perf_counter
        self._pipeline = None
        self._init_latency_ms = None
        self._identity_note = None

    @staticmethod
    def _default_importer():
        from paddleocr import PaddleOCRVL  # Kaggle-side only, lazy
        return PaddleOCRVL

    def ensure(self):
        """Return the cached pipeline, creating + timing it on first use."""
        if self._pipeline is None:
            started = self._clock()
            pipeline_cls = self._importer()
            # Authoritative v1.6 entry point; no version substitution.
            self._pipeline = pipeline_cls(pipeline_version=PIPELINE_VERSION)
            self._init_latency_ms = (self._clock() - started) * 1000.0
        return self._pipeline

    @property
    def init_latency_ms(self):
        """Model init/download time — reported separately, NEVER counted
        as per-document OCR latency."""
        return self._init_latency_ms

    @property
    def created(self):
        return self._pipeline is not None


def _coerce_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value)


def _coerce_grid(table):
    """Best-effort grid normalization; returns None when unusable."""
    if isinstance(table, str):  # e.g. raw HTML kept verbatim, one cell
        return [[table]]
    if isinstance(table, dict):
        for key in ("rows", "cells", "data", "grid"):
            if isinstance(table.get(key), list):
                table = table[key]
                break
        else:
            return None
    if not isinstance(table, list):
        return None
    grid = []
    for row in table:
        if isinstance(row, (list, tuple)):
            grid.append([str(c) for c in row])
        elif isinstance(row, dict):
            cells = row.get("cells", row.get("values"))
            if isinstance(cells, list):
                grid.append([str(c) for c in cells])
        elif isinstance(row, str):
            grid.append([row])
    return grid or None


def adapt_output(raw, doc_id=""):
    """Normalize structured pipeline output.

    Returns (text, fields, tables, visual_description, notes). Handles
    plain strings, Paddle-style dicts (rec_texts/rec_scores, blocks,
    tables, markdown, fields), and lists of blocks/strings. Unknown
    shapes are stringified into text + noted — never silently dropped,
    never hand-corrected.
    """
    texts, fields, tables, visuals, notes = [], {}, [], [], []
    if isinstance(raw, str):
        texts.append(raw)
        notes.append("shape: plain string")
        return "\n".join(texts), fields, tables, "", notes
    if isinstance(raw, (list, tuple)):
        if raw and all(isinstance(b, str) for b in raw):
            notes.append(f"shape: string list ×{len(raw)}")
            return "\n".join(raw), fields, tables, "", notes
        raw = {"blocks": list(raw)}
        notes.append("shape: block list -> blocks")
    if not isinstance(raw, dict):
        texts.append(str(raw))
        notes.append(f"shape: unrecognized {type(raw).__name__}; "
                     "stringified verbatim")
        return "\n".join(texts), fields, tables, "", notes

    for key in ("markdown", "md"):
        if isinstance(raw.get(key), str) and raw[key].strip():
            texts.append(raw[key])
            notes.append(f"used key: {key}")
    for key in ("text", "transcription", "content", "ocr_text"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value)
            notes.append(f"used key: {key}")
        elif isinstance(value, list) and value:
            texts.append(_coerce_text(value))
            notes.append(f"used key: {key} (list)")
    if isinstance(raw.get("rec_texts"), list) and raw["rec_texts"]:
        texts.append("\n".join(str(t) for t in raw["rec_texts"]))
        notes.append(f"used rec_texts ×{len(raw['rec_texts'])}")
        if "rec_scores" in raw:
            notes.append("rec_scores present (not scored, not in fields)")
    for key in ("blocks", "layout", "elements", "regions"):
        blocks = raw.get(key)
        if not isinstance(blocks, list) or not blocks:
            continue
        notes.append(f"used key: {key} ×{len(blocks)}")
        for block in blocks:
            if not isinstance(block, dict):
                texts.append(str(block))
                continue
            label = block.get("block_label", block.get("label",
                                                       block.get("type", "")))
            body = _coerce_text(block.get("text", block.get("content", "")))
            if body.strip():
                texts.append(f"[{label}] {body}" if label else body)
            for tkey in ("table", "tables", "cells", "rows", "html"):
                if tkey in block and block[tkey] is not None:
                    grid = _coerce_grid(block[tkey])
                    if grid:
                        tables.append(grid)
                        notes.append(f"block table via {tkey}")
    raw_tables = raw.get("tables", raw.get("table"))
    if raw_tables is not None:
        batch = raw_tables if isinstance(raw_tables, list) else [raw_tables]
        for table in batch:
            grid = _coerce_grid(table)
            if grid:
                tables.append(grid)
        notes.append("used key: tables/table")
    for key in ("fields", "key_values"):
        mapping = raw.get(key)
        if isinstance(mapping, dict):
            for fkey, fval in mapping.items():
                fields[str(fkey)] = fval if isinstance(fval, str) \
                    else str(fval)
            notes.append(f"used key: {key} ×{len(mapping)}")
    for key in ("chart", "figures", "chart_result", "visual_description",
                "description"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            visuals.append(value)
            notes.append(f"used key: {key}")
        elif isinstance(value, (list, dict)) and value:
            visuals.append(str(value))
            notes.append(f"used key: {key} (structured, stringified)")
    if not texts and not tables and not fields:
        notes.append("no recognized content keys; raw stringified to text")
        texts.append(str(raw))
    return "\n".join(texts), fields, tables, "\n".join(visuals), notes


def blocker_result(doc_id, reason, device=None):
    """Schema-valid result recording an import/runtime blocker (no crash)."""
    result = schema.blank_result(CANDIDATE, MODEL_ID,
                                 device or "unknown", doc_id)
    result["warnings"] = [reason]
    return result


def build_result(doc_id, raw, latency_ms, identity, raw_path=None,
                 notes_extra=None):
    """Schema-valid result from adapted pipeline output (no hand edits)."""
    text, fields, tables, visual, notes = adapt_output(raw, doc_id)
    if raw_path:
        notes.append(f"raw output preserved at: {raw_path}")
    if notes_extra:
        notes.extend(notes_extra)
    return {
        "model": identity.get("candidate", CANDIDATE),
        "model_version": identity.get("model_id", MODEL_ID),
        "device": identity.get("device", "unknown"),
        "document_id": doc_id,
        "latency_ms": latency_ms,
        "text": text,
        "fields": fields,
        "tables": tables,
        "visual_description": visual,
        "warnings": notes,
    }
