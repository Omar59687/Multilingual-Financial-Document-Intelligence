"""Standard experiment result contract (evaluation-side, stdlib only).

Every candidate approach (traditional OCR, PaddleOCR-VL, Qwen-VL, ColQwen
retrieval, …) emits this shape so one scorer compares them all:

{
  "model": "tesseract",            # family / engine name
  "model_version": "5.4.1",        # exact version or checkpoint id
  "device": "cpu",                 # e.g. cpu, kaggle-T4, local-cuda
  "document_id": "DEV-004",        # benchmark document
  "latency_ms": 1234.5,            # end-to-end for this document
  "text": "...",                   # recovered verbatim text ("" if none)
  "fields": {...},                 # structured field reads {name: raw string}
  "tables": [...],                 # recovered tables [[rows of strings]]
  "visual_description": "...",     # interpretation (charts/KPIs; "" if n/a)
  "warnings": [...]                # degradation/uncertainty notes
}

Extra keys are allowed; missing/wrongly-typed required keys are errors.
"""

REQUIRED_FIELDS = {
    "model": str,
    "model_version": str,
    "device": str,
    "document_id": str,
    "latency_ms": (int, float),
    "text": str,
    "fields": dict,
    "tables": list,
    "visual_description": str,
    "warnings": list,
}


def blank_result(model: str, model_version: str, device: str,
                 document_id: str) -> dict:
    """Schema-conformant empty result for experiment harnesses to fill."""
    return {
        "model": model,
        "model_version": model_version,
        "device": device,
        "document_id": document_id,
        "latency_ms": 0.0,
        "text": "",
        "fields": {},
        "tables": [],
        "visual_description": "",
        "warnings": [],
    }


def validate_result(result: dict) -> list:
    """Return a list of schema violations (empty = valid)."""
    errors = []
    if not isinstance(result, dict):
        return ["result must be a JSON object"]
    for key, types in REQUIRED_FIELDS.items():
        if key not in result:
            errors.append(f"missing required key: {key}")
        elif not isinstance(result[key], types):
            errors.append(
                f"key {key!r} must be {types}, "
                f"got {type(result[key]).__name__}")
    if isinstance(result.get("document_id"), str) \
            and not result["document_id"].strip():
        errors.append("document_id must be non-empty")
    latency = result.get("latency_ms")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        pass  # already reported above
    elif latency < 0:
        errors.append("latency_ms must be >= 0")
    return errors
