"""Qwen3-VL benchmark adapter (evaluation-side, stdlib only).

Covers the Kaggle-side Qwen contract WITHOUT importing transformers here:

- identity constants: 8B primary, 4B fallback-ONLY (see fallback rule),
- select_model_id: explicit 8B -> 4B fallback recording (never silent),
- extract_json_fields: safe parsing of a ```json block in generated
  prose — new {"pairs": [{"label","value"}]} contract plus legacy flat
  maps ({} when absent/truncated/unparseable),
- build_result / blocker_result: schema-conformant construction.

Fallback rule (locked): Qwen/Qwen3-VL-8B-Instruct runs first. The 4B
checkpoint (Qwen/Qwen3-VL-4B-Instruct) is used ONLY when 8B cannot
reasonably run on the Kaggle T4 (CUDA OOM, unsupported combination, or a
practical memory ceiling). The blocker is recorded explicitly in
warnings/meta; silent substitution is forbidden. If quantization becomes
necessary, STOP and report it as a decision/blocker instead.

No transformers import, no weights, no inference here — loading happens
only inside the Kaggle notebook via official Transformers patterns.
"""

import json

from . import schema

CANDIDATE = "qwen3-vl"
PRIMARY_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
FALLBACK_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def select_model_id(blocker_8b=None):
    """Choose the Qwen checkpoint, recording fallback explicitly.

    blocker_8b: None (8B runs) or a short reason string (e.g. "CUDA OOM
    on T4", "unsupported combination", "memory ceiling"). Returns
    (model_id, notes): 8B with a no-fallback note, or 4B with the blocker
    quoted in every note so substitution is never silent.
    """
    if not blocker_8b:
        return PRIMARY_MODEL_ID, ["qwen: 8B primary selected, no fallback"]
    reason = str(blocker_8b)
    return FALLBACK_MODEL_ID, [
        f"qwen: 8B blocked ({reason}); 4B fallback used",
        "qwen fallback rule: 4B ONLY because 8B could not run; "
        "recorded explicitly, never silent",
    ]


def identity_for(transformers_version=None, device=None,
                 model_id=None):
    """Reproducibility identity stamped on every Qwen result/meta."""
    chosen = model_id or PRIMARY_MODEL_ID
    return {
        "candidate": CANDIDATE,
        "model_id": chosen,
        "primary_model_id": PRIMARY_MODEL_ID,
        "fallback_model_id": FALLBACK_MODEL_ID,
        "is_fallback": chosen == FALLBACK_MODEL_ID,
        "transformers_version": transformers_version or "unknown",
        "device": device or "unknown",
    }


def extract_json_fields(generated_text):
    """Safely extract structured pairs from generated prose.

    Accepts two shapes (never raises, never invents):
    - new ``{"pairs": [{"label": ..., "value": ...}, ...]}`` contract:
      real labels are DATA (never object keys); order preserved,
      Unicode preserved, first occurrence wins on duplicate labels;
    - legacy flat ``{"label": "value", ...}`` maps (backward compatible).
    Returns {} when absent, truncated, or unparseable.
    Only flat string-valued pairs are kept; nested structures are
    stringified so the schema fields contract ({name: raw string}) holds.
    """
    parsed, _ = _extract_json_block(generated_text)
    if parsed is None:
        return {}
    fields, _, _ = pairs_to_fields(parsed)
    return fields


def _extract_json_block(generated_text):
    """Return (parsed_dict_or_None, state).

    State is "ok", "absent" (no fenced block), "truncated" (fence opened
    but never closed, e.g. generation hit the token cap), or "malformed".
    Truncated/malformed blocks yield None and must never fabricate pairs.
    """
    if not isinstance(generated_text, str) or "{" not in generated_text:
        return None, "absent"
    start_markers = ("```json", "```JSON", "```")
    text = generated_text
    fenced = False
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            fenced = True
            text = text[idx + len(marker):]
            end = text.find("```")
            if end == -1:
                return None, "truncated"
            text = text[:end]
            break
    if not fenced:
        return None, "absent"
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "malformed"
    try:
        parsed = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None, "malformed"
    if not isinstance(parsed, dict):
        return None, "malformed"
    return parsed, "ok"


def pairs_to_fields(parsed):
    """Convert a parsed JSON block into (fields, grid, notes).

    - ``{"pairs": [{"label","value"}, ...]}``: one field per pair, order
      preserved; duplicate labels keep the FIRST occurrence (recorded in
      notes, never silently overwritten); grid is one [label, value] row
      per kept pair (a structure the association scorer already accepts).
    - legacy flat maps: kept verbatim as before (stringified values).
    """
    fields, grid, notes = {}, [], []
    if not isinstance(parsed, dict):
        return fields, grid, ["json block not an object; ignored"]
    raw_pairs = parsed.get("pairs")
    if isinstance(raw_pairs, list):
        for entry in raw_pairs:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label", entry.get("name", entry.get("key")))
            value = entry.get("value", entry.get("val",
                                                 entry.get("amount")))
            label = "" if label is None else (
                label if isinstance(label, str) else str(label)).strip()
            value = "" if value is None else (
                value if isinstance(value, str) else str(value)).strip()
            if not label or not value:
                continue
            if label in fields:
                notes.append(f"duplicate label kept first: {label!r}")
                continue
            fields[label] = value
            grid.append([label, value])
        notes.append(f"pairs schema: {len(grid)} pairs kept")
        return fields, grid, notes
    for key, value in parsed.items():
        out_val = value if isinstance(value, str) else str(value)
        if str(key) in fields:
            notes.append(f"duplicate label kept first: {str(key)!r}")
            continue
        fields[str(key)] = out_val
    for label, value in fields.items():
        grid.append([label, value])
    return fields, grid, notes


def split_prose_visual(generated_text):
    """Split generated prose into (text, visual_description).

    Recognizes headings the model actually emits — "step 2" (legacy),
    "trend description:" and "visual description:" (case-insensitive) —
    whichever comes first. No heading: whole text is prose, visual is "".
    """
    text = generated_text if isinstance(generated_text, str) else ""
    lowered = text.lower()
    cut = -1
    for heading in ("step 2", "trend description:",
                    "visual description:"):
        idx = lowered.find(heading)
        if idx != -1 and (cut == -1 or idx < cut):
            cut = idx
    if cut == -1:
        return text.strip(), ""
    return text[:cut].strip(), text[cut:].strip()


def blocker_result(doc_id, reason, attempted_model_id=None, device=None):
    """Schema-valid result recording a Qwen blocker (no crash)."""
    result = schema.blank_result(CANDIDATE,
                                 attempted_model_id or PRIMARY_MODEL_ID,
                                 device or "unknown", doc_id)
    result["warnings"] = [reason]
    return result


def build_result(doc_id, text, visual_description, fields, tables,
                 latency_ms, identity, raw_path=None, notes_extra=None):
    """Schema-valid Qwen result from prompted outputs (no hand edits)."""
    notes = []
    if isinstance(identity, dict) and identity.get("is_fallback"):
        notes.append(
            f"qwen fallback active: {identity.get('model_id')} "
            f"(primary {identity.get('primary_model_id')} blocked; "
            "see experiment warnings)")
    if raw_path:
        notes.append(f"raw output preserved at: {raw_path}")
    if notes_extra:
        notes.extend(notes_extra)
    ident = identity or {}
    return {
        "model": ident.get("candidate", CANDIDATE),
        "model_version": ident.get("model_id", PRIMARY_MODEL_ID),
        "device": ident.get("device", "unknown"),
        "document_id": doc_id,
        "latency_ms": latency_ms,
        "text": text or "",
        "fields": dict(fields or {}),
        "tables": list(tables or []),
        "visual_description": visual_description or "",
        "warnings": notes,
    }
