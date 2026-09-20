"""Deterministic file router: extension map + magic-byte verification.

Signature wins over extension. Unknown types produce a controlled
IngestionStatus.UNSUPPORTED result (never an exception).
"""

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .model import FileType, IngestionStatus

EXTENSION_MAP = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".xlsx": FileType.XLSX,
    ".csv": FileType.CSV,
    ".png": FileType.PNG,
    ".jpg": FileType.JPG,
    ".jpeg": FileType.JPG,
}

SUPPORTED_SUFFIXES = set(EXTENSION_MAP) | {".json"}  # json: skipped sidecars


@dataclass
class Route:
    file_type: Optional[FileType]
    status: Optional[IngestionStatus]  # set only when UNSUPPORTED
    reason: str = ""


def _sniff_signature(head: bytes, path: Path) -> Optional[FileType]:
    if head.startswith(b"%PDF"):
        return FileType.PDF
    if head.startswith(b"\x89PNG"):
        return FileType.PNG
    if head.startswith(b"\xff\xd8\xff"):
        return FileType.JPG
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as zf:
                content_types = zf.read("[Content_Types].xml").decode(
                    "utf-8", errors="ignore")
        except Exception:
            return None
        if "wordprocessingml" in content_types:
            return FileType.DOCX
        if "spreadsheetml" in content_types:
            return FileType.XLSX
        return None
    return None


def route(path: Path) -> Route:
    """Identify a file's type. Reads only a small header (+ zip central
    directory for office formats)."""
    path = Path(path)
    if not path.is_file():
        return Route(None, IngestionStatus.FAILED, "missing file")
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError as exc:
        return Route(None, IngestionStatus.FAILED, f"unreadable: {exc}")
    sniffed = _sniff_signature(head, path)
    if sniffed is not None:
        return Route(sniffed, None, "signature")
    ext = path.suffix.lower()
    if ext == ".csv":
        # CSV has no magic: extension + UTF-8(S) decodability of a sample.
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                f.read(4096)
        except (OSError, UnicodeDecodeError) as exc:
            return Route(None, IngestionStatus.FAILED, f"unreadable csv: {exc}")
        return Route(FileType.CSV, None, "extension+decodable")
    if ext in EXTENSION_MAP:
        # Known extension but unrecognized signature (e.g. corrupt/empty).
        return Route(EXTENSION_MAP[ext], None, "extension-fallback")
    return Route(None, IngestionStatus.UNSUPPORTED,
                 f"unsupported type: {ext or '(no extension)'}")
