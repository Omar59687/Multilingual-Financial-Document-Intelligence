"""Build semantic vectors for the R1 chunk index (thin CLI, Phase 4 R3).

Usage (from repo root):
    python scripts/build_vectors.py
    python scripts/build_vectors.py --out-dir data/retrieval

Reads ONLY data/retrieval/chunks.json (never ground_truth/ or
canonical/ content). Writes vectors.json + vectors_meta.json under the
output directory (makedirs as needed). Updates index_meta.json with the
required embedding model identity and revision per docs/RETRIEVAL_DESIGN.md
section 8 while preserving all R1 chunker and corpus metadata. Never
modifies chunks.json. Prints vector count, dims, and encode time.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval.embeddings import (BATCH_SIZE, DEVICE, MODEL_ID,  # noqa: E402
                                  encode_texts, get_dependency_versions,
                                  load_model, resolve_revision)

# Basenames this CLI must never overwrite as raw output targets (owned by the
# R1 builder or the repo itself). Outputs are restricted to bare filenames
# resolved strictly under the output directory.
_PROTECTED_BASENAMES = frozenset(
    {"chunks.json", "index_meta.json", "requirements.txt"})


def update_index_meta(index_meta_path, model_id, repo, revision, dims=None,
                      vector_count=None, encode_ms=None):
    """Update index_meta.json in-place with embedding model metadata.

    Preserves all R1 chunker and corpus metadata, manifest hash,
    and existing counts.
    """
    path = Path(index_meta_path)
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    data["embedding_model"] = {
        "id": model_id,
        "repo": repo,
        "revision": revision,
        "revision_pinned": True,
        "note": ("Provisional per docs/RETRIEVAL_DESIGN.md section 6; "
                 "NOT a hard lock; swap path open."),
    }
    if vector_count is not None or dims is not None:
        if "counts" not in data or not isinstance(data["counts"], dict):
            data["counts"] = {}
        if vector_count is not None:
            data["counts"]["vectors_total"] = vector_count
        if dims is not None:
            data["counts"]["dims"] = dims
    if encode_ms is not None:
        if "timings_ms" not in data or not isinstance(data["timings_ms"], dict):
            data["timings_ms"] = {}
        data["timings_ms"]["vector_encode"] = encode_ms
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return True


def _resolve_output(raw, default_name, out_dir):
    """Resolve an output to ``out_dir / <bare basename>`` or reject it.

    Anything that is not a bare filename (directory components,
    absolute paths, traversal), any protected basename (compared
    case-insensitively, since Windows filesystems are
    case-insensitive), and anything escaping ``out_dir`` raises
    ValueError.
    """
    name = default_name if raw is None else raw
    if not isinstance(name, str) or not name:
        raise ValueError("output path must be a non-empty filename, "
                         "got %r" % (raw,))
    if Path(name).name != name:
        raise ValueError("output must be a bare filename under the "
                         "output directory, got %r" % (raw,))
    if name.casefold() in _PROTECTED_BASENAMES:
        raise ValueError("refusing to overwrite protected file %r" % (name,))
    resolved = out_dir / name
    if resolved.resolve().parent != out_dir.resolve():
        raise ValueError("output escapes the output directory: %r" % (raw,))
    return resolved


def _resolve_chunks(raw, out_dir):
    """Resolve the chunks input or reject prohibited corpora.

    Defaults to ``out_dir / "chunks.json"``. Inputs resolving inside
    the repo are permitted ONLY under ``data/retrieval/`` — the
    ground_truth/, canonical/ and eval corpora are never ingested, so
    a repo-relative input anywhere else raises ValueError. Inputs
    outside the repo (e.g. scratch dirs) are passed through.
    """
    path = Path(raw) if raw is not None else out_dir / "chunks.json"
    root = ROOT.resolve()
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        return path
    try:
        rel.relative_to(Path("data") / "retrieval")
    except ValueError:
        raise ValueError(
            "chunks input must live under data/retrieval/ (got %r); "
            "ground_truth/, canonical/ and eval corpora are never "
            "ingested" % (raw if raw is not None else str(path),))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default=None,
                        help="chunks.json path (default: data/retrieval/)")
    parser.add_argument("--out-dir", default=None,
                        help="vector output directory (git-ignored)")
    parser.add_argument("--vectors", default=None,
                        help="vectors.json path (default: <out-dir>/)")
    parser.add_argument("--meta", default=None,
                        help="vectors_meta.json path (default: <out-dir>/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data" / "retrieval"
    try:
        vectors_path = _resolve_output(args.vectors, "vectors.json",
                                       out_dir)
        meta_path = _resolve_output(args.meta, "vectors_meta.json",
                                    out_dir)
        chunks_path = _resolve_chunks(args.chunks, out_dir)
    except ValueError as exc:
        print("error: %s" % (exc,), file=sys.stderr)
        return 2
    if vectors_path.resolve() == meta_path.resolve():
        print("error: --vectors and --meta must be distinct files",
              file=sys.stderr)
        return 2
    if chunks_path.resolve() in (vectors_path.resolve(),
                                 meta_path.resolve()):
        print("error: --chunks input must not equal an output path",
              file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    total_started = time.perf_counter()

    load_started = time.perf_counter()
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise TypeError("chunks.json must hold a JSON list")
    load_ms = (time.perf_counter() - load_started) * 1000.0

    repo = (MODEL_ID if "/" in MODEL_ID
            else "sentence-transformers/" + MODEL_ID)

    rev_started = time.perf_counter()
    revision = resolve_revision()
    rev_ms = (time.perf_counter() - rev_started) * 1000.0

    model_started = time.perf_counter()
    model = load_model(revision=revision)
    model_ms = (time.perf_counter() - model_started) * 1000.0

    texts = [c["text"] for c in chunks]
    encode_started = time.perf_counter()
    vec_lists = encode_texts(texts, model=model)
    encode_ms = (time.perf_counter() - encode_started) * 1000.0

    vectors = {c["chunk_id"]: v for c, v in zip(chunks, vec_lists)}
    dims = len(vec_lists[0]) if vec_lists else 0

    write_started = time.perf_counter()
    vectors_path.write_text(
        json.dumps(vectors, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {
        "embedding_model": {
            "id": MODEL_ID,
            "repo": repo,
            "revision": revision,
            "revision_pinned": True,
            "note": ("Provisional per docs/RETRIEVAL_DESIGN.md section 6; "
                     "NOT a hard lock; swap path open. The revision above "
                     "was resolved live and passed to the loader, so the "
                     "recorded hash and the loaded weights match."),
        },
        "counts": {
            "chunks": len(chunks),
            "vectors": len(vectors),
            "dims": dims,
        },
        "encode": {
            "batch_size": BATCH_SIZE,
            "device": DEVICE,
            "eval_mode": True,
            "eval_verified": True,
            "normalized": True,
            "similarity": "cosine (= dot of L2-normalized vectors), "
                          "brute-force",
            "truncation": ("library tokenizer deterministic truncation; "
                           "chunk text used verbatim, never padded/invented"),
        },
        "dependencies": dict(get_dependency_versions(),
                             python=platform.python_version()),
        "corpus": {
            "chunks_path": str(chunks_path),
            "chunks_count": len(chunks),
        },
        "timings_ms": {
            "load_chunks": load_ms,
            "resolve_revision": rev_ms,
            "load_model": model_ms,
            "encode": encode_ms,
            "write": (time.perf_counter() - write_started) * 1000.0,
            "total": (time.perf_counter() - total_started) * 1000.0,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    index_meta_path = out_dir / "index_meta.json"
    if not index_meta_path.is_file() and chunks_path.parent != out_dir:
        alt = chunks_path.parent / "index_meta.json"
        if alt.is_file():
            index_meta_path = alt

    if index_meta_path.is_file():
        update_index_meta(
            index_meta_path,
            model_id=MODEL_ID,
            repo=repo,
            revision=revision,
            dims=dims,
            vector_count=len(vectors),
            encode_ms=encode_ms,
        )
        print("updated index metadata in %s" % (index_meta_path,))

    print("wrote %s (%d vectors, dim %d) and %s"
          % (vectors_path, len(vectors), dims, meta_path))
    print("model: %s revision %s" % (MODEL_ID, revision))
    print("encode time: %.1f ms (%.2f s)" % (encode_ms, encode_ms / 1000.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
