"""Trace E3 primary-target ranks without changing retrieval semantics.

This is a DEV-only diagnostic. It reuses the accepted E3 BM25 variant,
the frozen dense index, and the existing RRF implementation. It never
writes retrieval artifacts, changes the retriever, or reads EVAL gold.

Run from the repository root:
    python experiments/phase4_trace/trace_dev.py

The output is intentionally written under git-ignored data/ evidence.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _rank_map(ranking):
    return {item["chunk_id"]: item["rank"] for item in ranking}


def _metadata(record):
    """Keep trace output compact while retaining provenance clues."""
    keys = (
        "document_id", "filename", "file_type", "language", "page",
        "sheet", "kind", "section", "length", "element_ids",
    )
    return {key: record.get(key) for key in keys if key in record}


def _token_evidence(retriever, chunk_id):
    """Report tokenizer visibility when the pinned tokenizer is available."""
    record = retriever.get_chunk(chunk_id)
    text = record["text"]
    result = {"char_count": len(text), "token_count": None,
              "max_seq_length": None, "truncated": None}
    model = retriever._model
    tokenizer = getattr(model, "tokenizer", None) if model is not None else None
    if tokenizer is None:
        return result
    tokens = tokenizer(text, truncation=False)["input_ids"]
    limit = getattr(model, "max_seq_length", None)
    result["token_count"] = len(tokens)
    result["max_seq_length"] = limit
    result["truncated"] = bool(limit is not None and len(tokens) > limit)
    return result


def _hard_negative_flags(target, candidate):
    """Conservative labels; absence of a label is not proof of equivalence."""
    flags = []
    for field, label in (("document_id", "entity/document"),
                         ("page", "granularity/page"),
                         ("sheet", "granularity/sheet")):
        expected = target.get(field)
        observed = candidate.get(field)
        if expected is not None and observed is not None and expected != observed:
            flags.append(label)
    target_text = target.get("text", "").casefold()
    candidate_text = candidate.get("text", "").casefold()
    for marker, label in (("2024", "temporal"), ("2023", "temporal"),
                          ("2022", "temporal"), ("revenue", "metric"),
                          ("profit", "metric"), ("margin", "metric")):
        if marker in candidate_text and marker not in target_text:
            flags.append(label)
    return sorted(set(flags))


def main():
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from retrieval.hybrid import UnifiedRetriever, rrf_fuse
    from experiments.phase4_e3.identifier_recovery import ExperimentalBM25Index

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    retriever = UnifiedRetriever.from_artifacts()
    retriever._bm25 = ExperimentalBM25Index(list(retriever._chunks_by_id.values()))
    snapshot = snapshot_download(
        "sentence-transformers/" + retriever._model_id,
        revision=retriever._model_revision,
        local_files_only=True,
    )
    retriever._model = SentenceTransformer(
        snapshot, device="cpu", trust_remote_code=False
    ).eval()

    gold_path = ROOT / "data" / "eval" / "gold_questions.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    rows = []
    for question in gold["questions"]:
        query = question["text"]
        bm25 = retriever._full_bm25_ranking(query, None)
        dense = retriever._full_semantic_ranking(query, None)
        fused = rrf_fuse(bm25, dense, 50)
        bm25_ranks = _rank_map(bm25)
        dense_ranks = _rank_map(dense)
        fused_ranks = _rank_map(fused)
        primary = [target for target in question["targets"]
                   if target["grade"] == "primary"]
        target_rows = []
        for target in primary:
            cid = target["chunk_id"]
            record = retriever.get_chunk(cid)
            target_rows.append({
                "chunk_id": cid,
                "bm25_rank": bm25_ranks.get(cid),
                "dense_rank": dense_ranks.get(cid),
                "fused_rank": fused_ranks.get(cid),
                "in_bm25_top10": cid in {x["chunk_id"] for x in bm25[:10]},
                "in_dense_top10": cid in {x["chunk_id"] for x in dense[:10]},
                "in_fused_top10": cid in {x["chunk_id"] for x in fused[:10]},
                "metadata": _metadata(record),
                "tokenizer": _token_evidence(retriever, cid),
            })
        competitors = []
        target_records = [retriever.get_chunk(t["chunk_id"]) for t in primary]
        for hit in fused[:10]:
            if any(hit["chunk_id"] == t["chunk_id"] for t in primary):
                continue
            record = retriever.get_chunk(hit["chunk_id"])
            flags = sorted({flag for target in target_records
                            for flag in _hard_negative_flags(target, record)})
            competitors.append({"chunk_id": hit["chunk_id"],
                               "rank": hit["rank"],
                               "metadata": _metadata(record),
                               "hard_negative_flags": flags})
        rows.append({"qid": question["qid"], "lang": question["lang"],
                     "category": question["category"],
                     "targets": target_rows, "top10_competitors": competitors})

    output = {
        "status": "complete_dev_diagnostic",
        "control": "E3",
        "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "model": {"id": retriever._model_id,
                   "revision": retriever._model_revision},
        "rrf_k": 60,
        "trace_depth": 50,
        "questions": rows,
    }
    out = ROOT / "data" / "experiments" / "phase4-trace" / "dev_rank_trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({"output": str(out), "questions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
