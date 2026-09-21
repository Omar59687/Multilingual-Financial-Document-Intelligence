"""Export the Kaggle experiment bundle (git-ignored ZIP).

Usage (from repo root):
    python scripts/export_kaggle_bundle.py
    python scripts/export_kaggle_bundle.py --out data/benchmark/export/x.zip

Packages ONLY what a human needs to run the first benchmark on Kaggle:

    inputs/                     # 6 scoring images (copies/renders)
    manifest.json               # input provenance + render settings
    ocr_benchmark_truth.json    # derived scoring truth (evaluation-side)
    ocr_vision_benchmark.ipynb  # experiment notebook
    README_KAGGLE.md            # how to run (from notebooks/README.md)
    PROMPTS.md                  # controlled VLM prompts (from ocrbench)
    RESULT_SCHEMA.json          # result contract (from ocrbench)

NEVER included: canonical CSVs, full ground-truth JSONs, model weights,
secrets. Enforced by an allowlist + size caps + secret-pattern scan.
Do not commit generated ZIP files.
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocrbench import prompts, schema  # noqa: E402

BUNDLE_VERSION = "v0.1"
MAX_TOTAL_BYTES = 10 * 1024 * 1024
MAX_FILE_BYTES = 5 * 1024 * 1024
SECRET_PATTERNS = re.compile(
    r"(BEGIN [A-Z ]*PRIVATE KEY|aws_secret|AKIA[0-9A-Z]{16}|"
    r"api[_-]?key\s*[:=]\s*['\"][^'\"]{8,})", re.IGNORECASE)


def prompts_markdown() -> str:
    """Render the version-controlled prompt templates for the bundle."""
    lines = ["# Controlled VLM Prompts (benchmark evaluation only)",
             "",
             "Extraction and interpretation stay separate. Mirrors",
             "`src/ocrbench/prompts.py` (the version-controlled source).",
             ""]
    for name, text in prompts.PROMPTS.items():
        lines += [f"## {name}", "", text, ""]
    lines += ["## Prompts per document", ""]
    for doc, names in prompts.DOC_PROMPTS.items():
        lines.append(f"- {doc}: {', '.join(names)}")
    return "\n".join(lines) + "\n"


def result_schema_doc() -> dict:
    """Machine-readable result contract for the bundle."""
    return {
        "required": sorted(schema.REQUIRED_FIELDS),
        "types": {k: (t.__name__ if isinstance(t, type)
                      else "/".join(x.__name__ for x in t))
                  for k, t in schema.REQUIRED_FIELDS.items()},
        "notes": [
            "latency_ms >= 0; document_id non-empty.",
            "Extra keys allowed; missing/wrongly-typed keys are errors.",
            "Validated locally by src/ocrbench/schema.py.",
        ],
    }


def build_export(workspace_dir, notebooks_dir, out_zip) -> dict:
    """Assemble, scan, and write the bundle ZIP. Returns a report dict."""
    workspace_dir, notebooks_dir, out_zip = (
        Path(workspace_dir), Path(notebooks_dir), Path(out_zip))
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"mizaniq-ocr-benchmark-{BUNDLE_VERSION}"
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / prefix
        (stage / "inputs").mkdir(parents=True)
        manifest_src = workspace_dir / "manifest.json"
        manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
        for entry in manifest["entries"]:
            shutil.copy2(workspace_dir / entry["workspace_file"],
                         stage / entry["workspace_file"])
        for name in ("manifest.json", "ocr_benchmark_truth.json"):
            shutil.copy2(workspace_dir / name, stage / name)
        shutil.copy2(notebooks_dir / "ocr_vision_benchmark.ipynb",
                     stage / "ocr_vision_benchmark.ipynb")
        shutil.copy2(notebooks_dir / "README.md", stage / "README_KAGGLE.md")
        (stage / "PROMPTS.md").write_text(prompts_markdown(),
                                          encoding="utf-8")
        (stage / "RESULT_SCHEMA.json").write_text(
            json.dumps(result_schema_doc(), indent=2) + "\n",
            encoding="utf-8")
        # Safety gates before zipping.
        total, files = 0, []
        for path in sorted(stage.rglob("*")):
            if not path.is_file():
                continue
            size = path.stat().st_size
            assert size <= MAX_FILE_BYTES, f"oversize: {path.name}"
            total += size
            rel = path.relative_to(stage).as_posix()
            files.append(rel)
            if path.suffix.lower() in {".json", ".md", ".ipynb"}:
                content = path.read_text(encoding="utf-8")
                assert not SECRET_PATTERNS.search(content), \
                    f"secret-like pattern in {rel}"
        assert total <= MAX_TOTAL_BYTES, "bundle over size cap"
        with zipfile.ZipFile(out_zip, "w",
                             compression=zipfile.ZIP_DEFLATED) as zf:
            for rel in files:
                zf.write(stage / rel, f"{prefix}/{rel}")
    return {
        "zip": str(out_zip),
        "size_bytes": out_zip.stat().st_size,
        "files": files,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(
        ROOT / "data" / "benchmark" / "dataset_v0.1"))
    parser.add_argument("--notebooks", default=str(ROOT / "notebooks"))
    parser.add_argument("--out", default=str(
        ROOT / "data" / "benchmark" / "export"
        / f"mizaniq-ocr-benchmark-{BUNDLE_VERSION}.zip"))
    args = parser.parse_args()
    report = build_export(args.workspace, args.notebooks, args.out)
    print(f"bundle: {report['zip']} ({report['size_bytes'] / 1024:.0f} KB)")
    for name in report["files"]:
        print(f"  {name}")
    print("upload the ZIP contents to Kaggle; do not commit the ZIP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
