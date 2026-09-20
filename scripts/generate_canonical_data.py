"""Generate the validated MizanIQ canonical dataset_v0.1.

Usage (from repo root):
    python scripts/generate_canonical_data.py
    python scripts/generate_canonical_data.py --output data/canonical/dataset_v0.1

Pipeline: generate (fixed seed) -> write CSVs -> build manifest ->
validate on disk (fail-closed). Generated data is git-ignored; only
source code and docs are committed.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import config as C  # noqa: E402
from dataset import generator, validators  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "canonical" / C.DATASET_VERSION),
        help="output directory for dataset_v0.1",
    )
    args = parser.parse_args()
    out_dir = Path(args.output)

    tables = generator.generate_all(seed=C.RANDOM_SEED)
    generator.write_dataset(tables, out_dir)
    manifest = generator.build_manifest(tables, out_dir, seed=C.RANDOM_SEED)
    counts = validators.validate_all(out_dir)

    print(f"dataset: {manifest['dataset_version']}  seed: {manifest['random_seed']}")
    print(f"output:  {out_dir}")
    for name, count in counts.items():
        print(f"  {name}.csv: {count} rows")
    print("validation: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
