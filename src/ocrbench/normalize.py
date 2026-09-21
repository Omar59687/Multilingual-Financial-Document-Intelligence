"""Text normalization for OCR scoring (evaluation-side, stdlib only).

Verified empirically: NFKC folds NONE of the Arabic forms below, so every
rule here is an explicit, documented step (see docs/OCR_VISION_DESIGN.md).

Rules:
  1. NFKC (compatibility folding for Latin/presentation forms only).
  2. Arabic-Indic digits U+0660-0669 and Extended U+06F0-06F9 -> 0-9.
     (EVALUATION_PLAN §3: Indic digits normalized before comparison.)
  3. Arabic decimal separator U+066B -> '.', thousands U+066C removed.
  4. Tatweel U+0640 removed.
  5. Diacritics U+064B-0652 + U+0670 removed.
  6. Alef forms (U+0622, U+0623, U+0625, U+0671) -> bare alef U+0627.
  7. Whitespace runs collapsed to one space; outer whitespace stripped.

Deliberately NOT folded (would hide real errors): teh-marbuta/heh,
alef-maqsura/ya, waw/hamza-on-waw, punctuation (Arabic or Latin).
Digit errors ALWAYS survive: digits are compared exactly after folding.
"""

import re
import unicodedata
from decimal import Decimal, InvalidOperation

_INDIC_ZERO = ord("\u0660")
_EXT_INDIC_ZERO = ord("\u06F0")

_DIGIT_MAP = {chr(_INDIC_ZERO + i): str(i) for i in range(10)}
_DIGIT_MAP.update({chr(_EXT_INDIC_ZERO + i): str(i) for i in range(10)})
_DIGIT_RE = re.compile("|".join(_DIGIT_MAP))

_ALEF_MAP = {
    "\u0622": "\u0627",  # alef with madda
    "\u0623": "\u0627",  # alef with hamza above
    "\u0625": "\u0627",  # alef with hamza below
    "\u0671": "\u0627",  # alef wasla
}
_ALEF_RE = re.compile("|".join(_ALEF_MAP))

_DIACRITICS_RE = re.compile("[\u064b-\u0652\u0670]")
_TATWEEL = "\u0640"
_WS_RE = re.compile(r"\s+")

# Currency tokens stripped ONLY by normalize_amount (never by normalize_text,
# where "SAR" vs "ر.س" presence is itself a transcription signal).
_CURRENCY_RE = re.compile(r"(SAR|sar|SAR\.|ر\.س|ر\s*س)", re.IGNORECASE)


def fold_digits(text: str) -> str:
    """Map Arabic-Indic / Extended digits to Western 0-9 (exact otherwise)."""
    return _DIGIT_RE.sub(lambda m: _DIGIT_MAP[m.group(0)], text)


def normalize_text(text: str) -> str:
    """Apply rules 1,4,5,6,7 (+ digit fold 2, decimal/thousands 3 for
    separators only where unambiguous). Financial digit errors survive."""
    if text is None:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = fold_digits(out)
    out = out.replace("\u066c", "")          # Arabic thousands separator
    out = out.replace("\u066b", ".")         # Arabic decimal separator
    out = out.replace(_TATWEEL, "")
    out = _DIACRITICS_RE.sub("", out)
    out = _ALEF_RE.sub(lambda m: _ALEF_MAP[m.group(0)], out)
    out = _WS_RE.sub(" ", out).strip()
    return out


def normalize_amount(raw) -> Decimal:
    """Parse a financial amount string to Decimal quantized at 2dp.

    Accepts thousands separators (','), whitespace, SAR/ر.س tokens, and
    Arabic-Indic digits. Raises ValueError when unparseable — an
    unreadable amount is a miss, never a silent zero.
    """
    if isinstance(raw, (int, float, Decimal)):
        return Decimal(str(raw)).quantize(Decimal("0.01"))
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"unparseable amount: {raw!r}")
    out = unicodedata.normalize("NFKC", raw.strip())
    out = fold_digits(out)
    out = out.replace("\u066c", "").replace("\u066b", ".")
    out = _CURRENCY_RE.sub("", out)
    out = out.replace(",", "").replace(" ", "").replace("\u00a0", "")
    try:
        return Decimal(out).quantize(Decimal("0.01"))
    except InvalidOperation:
        raise ValueError(f"unparseable amount: {raw!r}")
