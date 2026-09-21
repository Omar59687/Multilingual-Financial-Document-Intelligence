"""Controlled VLM experiment prompts (evaluation-side).

Task-specific templates keep extraction and interpretation separate.
These are EVALUATION prompts for the benchmark — not production prompts.
The Kaggle notebook mirrors these strings so offline runs stay identical;
this module is the version-controlled source of truth.
"""

OCR_TRANSCRIBE = (
    "Transcribe all visible text faithfully, top to bottom. "
    "Preserve numbers, identifiers, and punctuation exactly as shown. "
    "For Arabic text, output the logical reading order. "
    "Do not infer, translate, or fill in missing values."
)

TABLE_RECOVER = (
    "Recover each visible table as structured rows. "
    "Associate every row label with its visible numeric value exactly "
    "as printed. Preserve numbers and identifiers character-for-character. "
    "Do not calculate, round, or invent missing cells. "
    "If a cell is unreadable, write UNREADABLE for that cell only."
)

CHART_READ = (
    "Step 1 — report only what is visible: chart title, axis labels, "
    "every labeled value, and the legend. "
    "Step 2 — separately, describe the trend using only the visible "
    "values (direction, notable shifts, highest/lowest point). "
    "Do not estimate values that are not labeled."
)

KPI_READ = (
    "Recover each displayed KPI card label and its value exactly as shown, "
    "then each bar label and value, then the budget panel figures. "
    "Separately, quote the visible budget-status statement verbatim. "
    "Do not recompute variances or reinterpret the verdict."
)

PROMPTS = {
    "ocr": OCR_TRANSCRIBE,
    "table": TABLE_RECOVER,
    "chart": CHART_READ,
    "kpi": KPI_READ,
}

# Which prompt(s) each benchmark document's experiment run uses.
DOC_PROMPTS = {
    "DEV-004": ["ocr"],
    "DEV-008": ["ocr", "chart"],
    "DEV-009": ["ocr", "kpi"],
    "DEV-010": ["ocr", "table"],
    "DEV-002": ["ocr", "table"],
    "DEV-003": ["ocr", "table"],
}
