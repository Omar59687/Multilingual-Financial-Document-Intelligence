"""R6 frozen retrieval measurement (evaluation harness, thin CLI, Phase 4 R6).

Runs the frozen G2 protocol per docs/RETRIEVAL_DESIGN.md section 10:
the same 12 DEV + 12 held-out EVAL questions, retrieval top_k=10, the
same frozen gold for all three modes (bm25/semantic/hybrid) through the
accepted R4 unified interface, scored with evaluation.g2_scoring
(Recall@10/Recall@5 primary + NDCG@10 graded, per-language slices +
minimum, latency p50/p95 per mode).

Split discipline: each gold file is loaded, measured, and reported
independently; splits are never pooled. Gold chunk IDs are scoring-only
(never passed to retrieval). Nothing is tuned: BM25/RRF constants are
frozen and measurement updates no parameters, so the held-out EVAL run
cannot leak into any model — EVAL numbers are reporting-only and any
future tuning must still use DEV alone.

Gold status: G1 human verification has NOT been performed, so every
gold item is unverified. Per EVAL section 2 and the gold-file R6
instruction, reported aggregates exclude all unverified items (they
report null with reason "pending G1"); question-level hits and
provisional scores are preserved flagged for audit. Thresholds are NOT
decided here (TBD -> G2 architect gate).

Usage (repo root):  python scripts/measure_retrieval.py

Writes data/eval/retrieval_g2_dev.json + retrieval_g2_eval.json
(git-ignored generated evidence) and prints a console summary.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import g2_scoring as G2  # noqa: E402
from retrieval.hybrid import UnifiedRetriever  # noqa: E402

GOLD_FILES = {
    "dev": ROOT / "data" / "eval" / "gold_questions.json",
    "eval": ROOT / "data" / "eval" / "gold_eval_questions.json",
}
DEFAULT_OUT_DIR = ROOT / "data" / "eval"


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_gold(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return doc["questions"], doc["_header"]


def _run_split(split, gold_path, retriever):
    questions, header = _load_gold(gold_path)
    verification = header.get("verification", {})
    evidence = G2.measure_split(retriever, questions, verification)
    reports = G2.build_mode_reports(evidence, verified_only=True)
    verified_count = sum(1 for q in evidence["questions"] if q["verified"])
    return {
        "split": split,
        "status": ("PROVISIONAL - gold unverified, pending G1 human check; "
                   "reported aggregates exclude unverified items"),
        "gold_file": str(gold_path),
        "gold_sha256": _sha256_file(gold_path),
        "n_questions": len(questions),
        "n_verified": verified_count,
        "evidence": evidence,
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", default=["dev", "eval"],
                        choices=["dev", "eval"],
                        help="splits to measure (default: dev eval)")
    parser.add_argument("--out-dir", default=None,
                        help="evidence output directory (default: data/eval)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    retriever = UnifiedRetriever.from_artifacts()
    meta = json.loads(
        (ROOT / "data" / "retrieval" / "index_meta.json").read_text(
            encoding="utf-8"))
    run_provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": meta["embedding_model"],
        "corpus_manifest_sha256": meta["corpus"]["manifest_sha256"],
        "corpus_counts": meta["counts"],
        "protocol": {"top_k": G2.K_TOP, "modes": list(G2.MODES),
                     "metrics": ["recall@5", "recall@10", "ndcg@10"],
                     "ndcg_gains": dict(G2.GRADE_GAINS),
                     "thresholds": "TBD - G2 architect gate"},
    }

    for split in args.splits:
        payload = _run_split(split, GOLD_FILES[split], retriever)
        payload["run"] = run_provenance
        out_path = out_dir / ("retrieval_g2_%s.json" % split)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print("split=%s questions=%d verified=%d -> %s"
              % (split, payload["n_questions"], payload["n_verified"],
                 out_path))
        for mode in G2.MODES:
            rep = payload["reports"][mode]
            lat = rep["latency"]
            print("  mode=%-8s latency_ms n=%d p50=%.1f p95=%.1f"
                  % (mode, lat["n"], lat["p50"], lat["p95"]))
            for metric in ("recall@5", "recall@10", "ndcg@10"):
                agg = rep["slices"]["overall"][metric]
                print("    %-9s verified-mean=%s (n=%d excluded=%d%s)" % (
                    metric, agg["mean"], agg["n"], agg["n_excluded"],
                    "; %s" % agg["reason"] if agg["reason"] else ""))
    print("STATUS: provisional - all gold unverified (G1 not performed); "
          "no gate decision made here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
