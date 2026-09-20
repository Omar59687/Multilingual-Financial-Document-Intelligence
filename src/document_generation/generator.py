"""Orchestration for DEV document generation (dataset_v0.1).

Pipeline: load canonical truth -> resolve deterministic selections ->
render 10 documents -> write ground-truth JSONs -> manifest -> validate
(fail-closed). Output layout::

    data/dev/dataset_v0.1/
      documents/      # the 10 DEV files ONLY (future ingestion corpus)
      ground_truth/   # DEV-001.json ... DEV-010.json (eval tooling, NOT corpus)
      manifest.json
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import documents, renderers, selection

DEV_VERSION = "dataset_v0.1"
MANIFEST_FILE = "manifest.json"

BUILDERS = (
    ("DEV-001", documents.build_dev001),
    ("DEV-002", documents.build_dev002),
    ("DEV-003", documents.build_dev003),
    ("DEV-004", documents.build_dev004),
    ("DEV-005", documents.build_dev005),
    ("DEV-006", documents.build_dev006),
    ("DEV-007", documents.build_dev007),
    ("DEV-008", documents.build_dev008),
    ("DEV-009", documents.build_dev009),
    ("DEV-010", documents.build_dev010),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_all(canonical_dir, out_dir) -> dict:
    """Render the full DEV set; returns dev_id -> ground-truth record."""
    canonical_dir = Path(canonical_dir)
    out_dir = Path(out_dir)
    docs_dir = out_dir / "documents"
    gt_dir = out_dir / "ground_truth"
    docs_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    tables = selection.load_canonical(canonical_dir)
    resolved = selection.resolve_all(tables)

    records = {}
    files = {}
    for dev_id, builder in BUILDERS:
        filename, record = builder(resolved[dev_id], docs_dir)
        records[dev_id] = record
        gt_path = gt_dir / f"{dev_id}.json"
        gt_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        files[filename] = {
            "dev_id": dev_id,
            "sha256": sha256_file(docs_dir / filename),
            "size_bytes": (docs_dir / filename).stat().st_size,
        }

    canonical_manifest = json.loads(
        (canonical_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "dev_version": DEV_VERSION,
        "random_seed": selection.SEED,
        "canonical_dataset": canonical_manifest.get("dataset_version"),
        "canonical_manifest_sha256": sha256_file(
            canonical_dir / "manifest.json"),
        "no_leakage_rule": ("ground_truth/ and canonical CSVs are evaluation/"
                            "build fixtures and live OUTSIDE the ingestible "
                            "corpus. Future ingestion may read ONLY "
                            "documents/. Ground-truth IDs/answers are never "
                            "embedded in document bytes beyond visible "
                            "content (e.g. the printed invoice ID)."),
        "generation_date_policy": ("generated_at is informational only and "
                                   "is excluded from determinism checks"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": files,
        "ground_truth_files": {
            f"{dev_id}.json": {
                "sha256": sha256_file(gt_dir / f"{dev_id}.json")}
            for dev_id in records
        },
    }
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return records
