"""Phase-2 benchmark preparation: export scoring inputs from frozen fixtures.

Usage (from repo root):
    python scripts/prepare_ocr_benchmark.py
    python scripts/prepare_ocr_benchmark.py --out data/benchmark/dataset_v0.1

Reads ONLY frozen dataset_v0.1 documents + ground truth (never modifies
originals). Writes a git-ignored workspace:

    data/benchmark/dataset_v0.1/
      inputs/                     # PNG/JPG scoring images
      manifest.json               # per-input provenance + render settings
      ocr_benchmark_truth.json    # derived scoring truth (evaluation-side)

Copies are byte-identical (sha256-verified). PDF pages are rendered with
pypdfium2 at fixed scale=2 (byte-deterministic; verified by test).
No OCR, no models, no weights here — inputs only.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocrbench.truth import BENCHMARK_DOCS, derive_all  # noqa: E402

RENDER_SCALE = 2  # pypdfium2 72dpi units -> ~144dpi PNG (fixed, documented)

# Workspace filename per benchmark document (copies keep native bytes;
# PDF pages render to PNG at RENDER_SCALE).
WORKSPACE_FILES = {
    "DEV-004": ("copy", "DEV-004.png"),
    "DEV-008": ("copy", "DEV-008.png"),
    "DEV-009": ("copy", "DEV-009.jpg"),
    "DEV-010": ("render", "DEV-010_p1.png"),
    "DEV-002": ("render", "DEV-002_p1.png"),
    "DEV-003": ("render", "DEV-003_p1.png"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_page(pdf_path: Path, page_index: int, out_path: Path) -> dict:
    """Render one PDF page to PNG (deterministic settings)."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page = doc[page_index]
        width_pt, height_pt = page.get_size()
        pil = page.render(scale=RENDER_SCALE).to_pil()
        pil.save(str(out_path), format="PNG")
    finally:
        doc.close()
    try:
        engine_ver = getattr(pdfium, "__version__", "pinned-5.6.0")
    except Exception:
        engine_ver = "pinned-5.6.0"
    return {
        "engine": f"pypdfium2-{engine_ver}",
        "scale": RENDER_SCALE,
        "page_index": page_index,
        "width_pt": round(float(width_pt), 2),
        "height_pt": round(float(height_pt), 2),
        "width_px": pil.size[0],
        "height_px": pil.size[1],
    }


def _image_size(path: Path) -> dict:
    from PIL import Image
    with Image.open(path) as img:
        img.load()
        return {"format": img.format, "width_px": img.width,
                "height_px": img.height, "mode": img.mode}


def build_workspace(docs_dir, gt_dir, out_dir) -> dict:
    """Export benchmark inputs + manifest + derived truth. Returns manifest."""
    docs_dir, gt_dir, out_dir = Path(docs_dir), Path(gt_dir), Path(out_dir)
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for dev_id, (role, tasks) in BENCHMARK_DOCS.items():
        matches = sorted(docs_dir.glob(f"{dev_id}*"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one {dev_id}* in {docs_dir}, "
                f"found {[p.name for p in matches]}")
        source = matches[0]
        kind, workspace_name = WORKSPACE_FILES[dev_id]
        dest = inputs_dir / workspace_name
        entry = {
            "document_id": dev_id,
            "source_filename": source.name,
            "source_sha256": _sha256(source),
            "workspace_file": f"inputs/{workspace_name}",
            "kind": kind,
            "role": role,
            "tasks": list(tasks),
        }
        if kind == "copy":
            with open(source, "rb") as fsrc, open(dest, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
            assert _sha256(dest) == entry["source_sha256"], \
                f"copy corrupted: {dev_id}"
            entry["image"] = _image_size(dest)
        else:
            entry["render"] = _render_page(source, 0, dest)
            entry["image"] = _image_size(dest)
        entry["workspace_sha256"] = _sha256(dest)
        entries.append(entry)
    manifest = {
        "dataset": "dataset_v0.1",
        "created_from": "frozen DEV documents (originals unmodified)",
        "render_scale": RENDER_SCALE,
        "entries": entries,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    truth = derive_all(gt_dir)
    (out_dir / "ocr_benchmark_truth.json").write_text(
        json.dumps(truth, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default=str(
        ROOT / "data" / "dev" / "dataset_v0.1" / "documents"))
    parser.add_argument("--gt", default=str(
        ROOT / "data" / "dev" / "dataset_v0.1" / "ground_truth"))
    parser.add_argument("--out", default=str(
        ROOT / "data" / "benchmark" / "dataset_v0.1"))
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = build_workspace(args.docs, args.gt, args.out)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(f"{'document':10} {'role':10} {'tasks':16} "
          f"{'kind':8} {'workspace_file'}")
    for entry in manifest["entries"]:
        print(f"{entry['document_id']:10} {entry['role']:10} "
              f"{','.join(entry['tasks']):16} {entry['kind']:8} "
              f"{entry['workspace_file']}")
    print(f"\nworkspace: {args.out} "
          f"({len(manifest['entries'])} inputs in {elapsed_ms:.0f} ms)")
    print("benchmark inputs ready — no models run, no weights downloaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
