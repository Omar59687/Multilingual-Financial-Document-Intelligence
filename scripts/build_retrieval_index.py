"""Build the R1 text-retrieval index (thin CLI over retrieval.chunking).

Usage (from repo root):
    python scripts/build_retrieval_index.py
    python scripts/build_retrieval_index.py --out-dir data/retrieval

Reads ONLY documents/ plus the Paddle OCR sidecar (never ground_truth/
or canonical CSVs). Writes chunks.json + index_meta.json under
data/retrieval/ (git-ignored) and prints per-doc chunk counts plus the
docs excluded (zero chunks) and any warnings. R1 carries no vectors.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval.chunking import build_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", default=None,
                        help="ingestible corpus directory (documents/ ONLY)")
    parser.add_argument("--ocr-path", default=None,
                        help="paddle_results.json OCR sidecar")
    parser.add_argument("--out-dir", default=None,
                        help="index output directory (git-ignored)")
    args = parser.parse_args()

    summary = build_index(docs_dir=args.docs_dir, ocr_path=args.ocr_path,
                          out_dir=args.out_dir)
    print(f"wrote {summary['chunks_path']} "
          f"({len(summary['chunks'])} chunks) and {summary['meta_path']}")
    for doc_id in sorted(summary["counts_by_doc"]):
        print(f"  {doc_id}: {summary['counts_by_doc'][doc_id]} chunks")
    print(f"docs excluded (zero chunks): "
          f"{summary['excluded'] or 'none'}")
    if summary["warnings"]:
        print("warnings:")
        for warning in summary["warnings"]:
            print(f"  {warning['file']}: {warning['warning']}")
    else:
        print("warnings: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
