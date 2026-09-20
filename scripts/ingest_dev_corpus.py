"""Phase-1 baseline: ingest the frozen DEV corpus, dump normalized JSON.

Usage (from repo root):
    python scripts/ingest_dev_corpus.py
    python scripts/ingest_dev_corpus.py --corpus data/dev/dataset_v0.1/documents

Reads ONLY documents/ (never ground_truth/ or canonical CSVs). Writes
per-document normalized JSON under data/ingestion/ (git-ignored) and prints
the baseline table: parser, status, elements, chars, tables/rows, latency,
OCR/visual flags. No quality scores are invented here.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingestion.model import ElementType  # noqa: E402
from ingestion.service import ingest_directory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        default=str(ROOT / "data" / "dev" / "dataset_v0.1" / "documents"),
        help="ingestible corpus directory (documents/ ONLY)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "ingestion" / "dataset_v0.1"),
        help="normalized JSON output directory (git-ignored)",
    )
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    report = ingest_directory(args.corpus)
    wall_ms = (time.perf_counter() - started) * 1000.0

    print(f"{'document':44} {'parser':16} {'status':15} "
          f"{'elems':>6} {'chars':>7} {'tbl/row':>9} {'ms':>8}  flags")
    for result in report["results"]:
        doc = result.document
        (out_dir / f"{doc.document_id}.json").write_text(
            doc.to_json() + "\n", encoding="utf-8")
        chars = sum(len(e.text or "") for e in doc.elements)
        tables = sum(1 for e in doc.elements
                     if e.element_type == ElementType.TABLE)
        rows = sum(1 for e in doc.elements if e.element_type in
                   (ElementType.CSV_ROW, ElementType.TABLE_ROW))
        if doc.file_type and doc.file_type.value == "xlsx":
            rows = doc.metadata.get("data_cells", rows)
        parser = doc.metadata.get("parser", "-")
        flags = ",".join(f for f, on in
                         (("ocr", doc.requires_ocr),
                          ("visual", doc.requires_visual)) if on) or "-"
        print(f"{doc.filename:44} {parser:16} {doc.status.value:15} "
              f"{len(doc.elements):6} {chars:7} {tables}/{rows:<7} "
              f"{doc.latency_ms:8.1f}  {flags}")

    summary = report["summary"]
    print(f"\ningested {summary['ingested']}/{summary['files_found']} "
          f"in {summary['total_latency_ms']:.1f} ms (wall {wall_ms:.1f} ms); "
          f"by_status={summary['by_status']}; "
          f"skipped={summary['skipped_sidecars']}; failed={summary['failed']}")
    (out_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    if summary["failed"]:
        print("BASELINE HAS FAILURES — investigate before proceeding")
        return 1
    print("baseline: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
