"""Stored post-G1 evidence only; additional coverage diagnostic, no retrieval."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/phase4-answer-coverage"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(split):
    gold_path = ROOT / "data/eval" / ("gold_questions.json" if split == "dev" else "gold_eval_questions.json")
    report_path = ROOT / "data/eval" / f"retrieval_g2_{split}.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["gold_sha256"] == sha(gold_path)
    questions = {q["qid"]: q for q in gold["questions"]}
    rows = []
    for evidence in report["evidence"]["questions"]:
        q = questions[evidence["qid"]]
        signoff = gold["_header"]["verification"][q["qid"]]
        assert signoff["verified"] is True and signoff["verified_by"].strip() and signoff["verified_date"].strip()
        assert evidence["verified"] is True and evidence["targets"] == q["targets"]
        primary = {t["chunk_id"] for t in q["targets"] if t["grade"] == "primary"}
        assert primary
        ids = evidence["per_mode"]["hybrid"]["retrieved_ids"]
        assert len(ids) == len(set(ids))
        row = {"qid": q["qid"], "n_primary": len(primary), "primary_ids": sorted(primary), "basis": q.get("basis", "")}
        for k in (5, 10):
            hits = sorted(primary.intersection(ids[:k]))
            row[f"retrieved_{k}"] = len(hits)
            row[f"hit_ids_{k}"] = hits
            row[f"strict_recall_{k}"] = len(hits) / len(primary)
            row[f"any_primary_hit_{k}"] = int(bool(hits))
        gains = {t["chunk_id"]: {"primary": 2, "acceptable": 1}[t["grade"]] for t in q["targets"]}
        dcg = sum(gains.get(cid, 0) / math.log2(rank + 1) for rank, cid in enumerate(ids[:10], 1))
        ideal = sum(g / math.log2(rank + 1) for rank, g in enumerate(sorted(gains.values(), reverse=True)[:10], 1))
        row["ndcg_10"] = dcg / ideal
        rows.append(row)
    assert len(rows) == 12 and len(rows) == len(questions)
    summary = {"n_verified": len(rows), "multi_primary_questions": sum(r["n_primary"] > 1 for r in rows)}
    for k in (5, 10):
        summary[f"strict_recall_{k}"] = sum(r[f"strict_recall_{k}"] for r in rows) / len(rows)
        summary[f"answer_coverage_{k}"] = sum(r[f"any_primary_hit_{k}"] for r in rows) / len(rows)
        summary[f"partial_answer_covered_{k}"] = sum(0 < r[f"retrieved_{k}"] < r["n_primary"] for r in rows)
        summary[f"true_failures_{k}"] = [r["qid"] for r in rows if not r[f"any_primary_hit_{k}"]]
        summary[f"coverage_minus_strict_{k}"] = summary[f"answer_coverage_{k}"] - summary[f"strict_recall_{k}"]
        assert abs(summary[f"strict_recall_{k}"] - report["reports"]["hybrid"]["slices"]["overall"][f"recall@{k}"]["mean"]) < 1e-12
    summary["ndcg_10"] = sum(r["ndcg_10"] for r in rows) / len(rows)
    assert abs(summary["ndcg_10"] - report["reports"]["hybrid"]["slices"]["overall"]["ndcg@10"]["mean"]) < 1e-12
    return {"source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (gold_path, report_path)}, "summary": summary, "questions": rows}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    result = {split: analyze(split) for split in ("dev", "eval")}
    (OUT / "coverage.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split, data in result.items():
        print(split, json.dumps(data["summary"]))
        for row in data["questions"]:
            print(row["qid"], row["n_primary"], row["retrieved_5"], row["retrieved_10"], row["strict_recall_5"], row["strict_recall_10"], row["any_primary_hit_5"], row["any_primary_hit_10"])
