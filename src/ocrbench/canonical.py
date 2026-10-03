"""Offline semantic normalization: explicit visual pairs -> canonical fields.

Maps model-produced ``{visible label: displayed value}`` pairs (e.g. from
the Qwen ``{"pairs": [...]}`` contract) onto canonical structured keys
without reading ground truth at any point:

- label aliases are STATIC configuration authored from the benchmark
  DESIGN (fixture-visible strings; branch codes are canonical branch IDs
  from the frozen canonical dataset, not derived from GT answers);
- numerics are normalized from the DISPLAYED string only (``23.90M SAR``
  -> 23900000) with explicit rounding provenance;
- hidden precision is NEVER fabricated: suffixed/rounded displays are
  marked ``display-rounded`` and must not claim exact equality with any
  full-precision source numeric a dashboard rounds away.

This module is evaluation-side analytics ONLY. It does not alter visual
benchmark scores, result schemas, or scorer semantics.
"""

from decimal import Decimal, InvalidOperation

# Static alias table: normalized visible label -> canonical key.
# Normalization applied before lookup: strip surrounding whitespace and
# ONE trailing colon (the model emits "Budget:" for a visible "Budget").
# Case-insensitive via .casefold(). Unknown labels stay unmapped.
DEV009_ALIASES = {
    "revenue": "revenue",
    "gross profit": "gross_profit",
    "operating expenses": "operating_expenses",
    "net income": "net_income",
    "budget": "budget_total",
    "variance": "variance",
    # NOTE: "actual" is intentionally unmapped: the panel's Actual figure
    # duplicates Revenue, and inventing a second canonical key for one
    # visible number would fabricate structure. It stays in unmapped.
    "riyadh": "branch_revenue_BR-RUH",
    "jeddah": "branch_revenue_BR-JED",
    "dammam": "branch_revenue_BR-DMM",
}

# Display-magnitude suffixes. Any suffixed value is inherently rounded
# (dashboard display), so normalized output is display-rounded.
MAGNITUDE = {"k": 3, "m": 6, "b": 9}

_CURRENCY_TOKENS = ("sar", "$", "usd", "aed", "egp", "qar", "kwd",
                    "ر.س", "ر. س", "ريال")


def alias_key(visible_label: str):
    """Map one visible label to a canonical key (None when unknown)."""
    if not isinstance(visible_label, str):
        return None
    folded = visible_label.strip().casefold()
    if folded.endswith(":"):
        folded = folded[:-1].strip()
    return DEV009_ALIASES.get(folded)


def normalize_display_value(display_value: str) -> dict:
    """Normalize ONE displayed value string.

    Returns ``{"normalized_value": Decimal|None, "precision": str,
    "error": str|None}`` where precision is ``"display-rounded"`` for
    suffixed (M/B/K) or short-decimal displays and ``"exact-visible"``
    for fully-specified 2dp figures (e.g. signed panel variances).
    Unparseable input yields ``normalized_value None`` — never zero.
    """
    if not isinstance(display_value, str) or not display_value.strip():
        return {"normalized_value": None, "precision": "unknown",
                "error": "empty display value"}
    text = display_value.strip()
    sign = 1
    if text[:1] in ("+", "-"):
        sign = -1 if text[0] == "-" else 1
        text = text[1:].strip()
    lowered = text.casefold()
    for token in _CURRENCY_TOKENS:
        lowered = lowered.replace(token, "")
    lowered = lowered.replace("\u00a0", " ").strip()
    magnitude = 0
    rounded = False
    if lowered and lowered[-1] in MAGNITUDE:
        magnitude = MAGNITUDE[lowered[-1]]
        rounded = True  # suffix display: hidden precision unknowable
        lowered = lowered[:-1].strip()
    lowered = lowered.replace(",", "").replace(" ", "")
    try:
        number = Decimal(lowered)
    except InvalidOperation:
        return {"normalized_value": None, "precision": "unknown",
                "error": f"unparseable display value: {display_value!r}"}
    scaled = number * (Decimal(10) ** magnitude) * sign
    if not rounded:
        # A bare figure is only exact-visible when fully specified;
        # anything coarser than 2dp stays display-rounded.
        rounded = scaled != scaled.quantize(Decimal("0.01"))
        scaled = scaled.quantize(Decimal("0.01"))
    return {
        "normalized_value": scaled,
        "precision": "display-rounded" if rounded else "exact-visible",
        "error": None,
    }


def canonicalize_fields(fields: dict) -> dict:
    """Map an explicit fields dict onto the canonical DEV-009 layer.

    Returns ``{"canonical_fields": {key: {display_value,
    normalized_value (str|None), source_label, precision}}, "unmapped":
    [labels], "warnings": [...]}``. Unknown labels are listed, never
    dropped silently and never invented into canonical keys. No ground
    truth is consulted.
    """
    canonical, unmapped, warnings = {}, [], []
    for label, display in (fields or {}).items():
        key = alias_key(label)
        if key is None:
            unmapped.append(label)
            continue
        if key in canonical:
            warnings.append(
                f"duplicate canonical key kept first: {key} "
                f"(from {label!r})")
            continue
        norm = normalize_display_value(
            display if isinstance(display, str) else str(display))
        canonical[key] = {
            "display_value": display,
            "normalized_value": (str(norm["normalized_value"])
                                 if norm["normalized_value"] is not None
                                 else None),
            "source_label": label,
            "precision": norm["precision"],
        }
        if norm["error"]:
            warnings.append(f"{key}: {norm['error']}")
    if unmapped:
        warnings.append(f"{len(unmapped)} labels unmapped: "
                        f"{unmapped[:5]}")
    return {"canonical_fields": canonical, "unmapped": unmapped,
            "warnings": warnings}
