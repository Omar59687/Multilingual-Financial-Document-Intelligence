"""Normalized document model for Phase-1 native ingestion.

One shape for all formats (dataclasses only — no new dependencies).
Flat elements with coordinates preserved for future chunking/citations.
Deterministic IDs throughout (no UUIDs); see document_id_for().
"""

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

DEV_ID_RE = re.compile(r"^(DEV-\d{3})")


class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    PNG = "png"
    JPG = "jpg"


class ElementType(str, Enum):
    TEXT = "text"               # paragraph / text block (page-anchored)
    HEADING = "heading"         # docx heading / PDF outline-ish title runs
    TABLE = "table"             # container: shape in meta, cells separate
    TABLE_ROW = "table_row"     # docx row convenience (cells also emitted)
    TABLE_CELL = "table_cell"   # page-anchored table cell
    SHEET_CELL = "sheet_cell"   # xlsx cell: native value + data_type
    CSV_ROW = "csv_row"         # csv data row: mapping + row_number
    IMAGE_REF = "image_ref"     # existence marker (no OCR in Phase 1)


class IngestionStatus(str, Enum):
    NATIVE_OK = "native_ok"             # all units natively sufficient
    PARTIAL_NATIVE = "partial_native"   # some units need OCR (requires_ocr)
    REQUIRES_OCR = "requires_ocr"       # scanned-style: native insufficient
    REQUIRES_VISUAL = "requires_visual"  # raster image: metadata only
    FAILED = "failed"                   # missing/corrupt/parser exception
    UNSUPPORTED = "unsupported"         # unknown type (router-level)


class LanguageHint(str, Enum):
    AR = "ar"
    EN = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class Element:
    element_id: str
    element_type: ElementType
    page: Optional[int] = None          # 1-based PDF/DOCX page (None if n/a)
    sheet: Optional[str] = None         # xlsx sheet name
    row: Optional[int] = None           # PDF/DOCX table cells: 0-based;
                                        # XLSX: native 1-based; CSV row_number: 1-based
    column: Optional[int] = None
    table_index: Optional[int] = None   # 0-based table order on page/sheet
    text: Optional[str] = None          # verbatim content, outer whitespace stripped
    value: Any = None                   # native value (xlsx numbers stay numbers)
    data_type: Optional[str] = None     # xlsx cell data_type / csv "str"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["element_type"] = self.element_type.value
        return d


@dataclass
class Document:
    document_id: str
    filename: str
    file_type: Optional[FileType]
    language_hint: LanguageHint = LanguageHint.UNKNOWN
    page_count: Optional[int] = None
    sheet_count: Optional[int] = None
    status: IngestionStatus = IngestionStatus.FAILED
    requires_ocr: bool = False
    requires_visual: bool = False
    metadata: dict = field(default_factory=dict)
    elements: list = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type.value if self.file_type else None,
            "language_hint": self.language_hint.value,
            "page_count": self.page_count,
            "sheet_count": self.sheet_count,
            "status": self.status.value,
            "requires_ocr": self.requires_ocr,
            "requires_visual": self.requires_visual,
            "metadata": self.metadata,
            "elements": [e.to_dict() for e in self.elements],
            "error": self.error,
            "latency_ms": self.latency_ms,
        }

    def to_json(self, **kwargs) -> str:
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)


@dataclass
class IngestedDocument:
    """Service envelope: never raises for domain failures."""
    document: Document
    ok: bool
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {"document": self.document.to_dict(), "ok": self.ok,
                "error": self.error, "latency_ms": self.latency_ms}


def document_id_for(path: Path, content_sha256: Optional[str] = None) -> str:
    """Stable IDs: frozen DEV files use their DEV-### prefix; everything else
    is content-addressed (DOC-<12 hex of sha256). Same bytes -> same ID."""
    match = DEV_ID_RE.match(path.name)
    if match:
        return match.group(1)
    digest = content_sha256 or hashlib.sha256(path.read_bytes()).hexdigest()
    return f"DOC-{digest[:12]}"
