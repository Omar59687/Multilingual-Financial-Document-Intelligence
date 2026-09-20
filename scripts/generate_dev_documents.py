"""Generate the validated MizanIQ DEV document set (dataset_v0.1).

Usage (from repo root):
    python scripts/generate_dev_documents.py
    python scripts/generate_dev_documents.py --output data/dev/dataset_v0.1
    python scripts/generate_dev_documents.py --rebuild-canonical

Pipeline: (optionally rebuild canonical truth) -> resolve deterministic
selections -> render 10 DEV documents -> ground-truth JSONs -> manifest ->
validate on disk (fail-closed). Everything under data/ is git-ignored.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import config as C  # noqa: E402
from dataset import generator as canonical_generator  # noqa: E402
from dataset import validators as canonical_validators  # noqa: E402
from document_generation import generator as dev_generator  # noqa: E402
from document_generation import validators as dev_validators  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        default=str(ROOT / "data" / "canonical" / C.DATASET_VERSION),
        help="canonical dataset_v0.1 directory",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "dev" / C.DATASET_VERSION),
        help="output directory for the DEV set",
    )
    parser.add_argument(
        "--rebuild-canonical", action="store_true",
        help="regenerate + validate canonical truth before rendering docs",
    )
    args = parser.parse_args()
    canonical_dir = Path(args.canonical)
    out_dir = Path(args.output)

    if args.rebuild_canonical or not (canonical_dir / "manifest.json").exists():
        tables = canonical_generator.generate_all(seed=C.RANDOM_SEED)
        canonical_generator.write_dataset(tables, canonical_dir)
        canonical_generator.build_manifest(tables, canonical_dir,
                                           seed=C.RANDOM_SEED)
        canonical_validators.validate_all(canonical_dir)
        print("canonical dataset_v0.1: regenerated + validated")

    dev_generator.generate_all(canonical_dir, out_dir)
    files = dev_validators.validate_all(canonical_dir, out_dir)

    print(f"dev set: dataset_v0.1  seed: {C.RANDOM_SEED}")
    print(f"output:  {out_dir}")
    for dev_id, filename in files.items():
        size = (out_dir / "documents" / filename).stat().st_size
        print(f"  {dev_id}: {filename} ({size / 1024:.1f} KB)")
    print("validation: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
