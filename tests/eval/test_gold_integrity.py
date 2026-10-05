"""Integrity tests for the frozen Phase-4 retrieval-eval gold set (package R5).

Scope: validates the SPLIT gold files — data/eval/gold_questions.json (dev,
Q-DEV-*) and data/eval/gold_eval_questions.json (held-out, Q-EVAL-*) —
WITHOUT executing any retriever. This module imports stdlib only
(json/pathlib/ast/re) and must never import retrieval machinery; that
property is itself asserted below. No dev-tooling loader helper exists in
product code yet, so held-out exclusivity is enforced structurally here:
per-file separation plus cross-contamination guards.
"""

import ast
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = REPO_ROOT / "data" / "eval" / "gold_questions.json"
EVAL_PATH = REPO_ROOT / "data" / "eval" / "gold_eval_questions.json"
CHUNKS_PATH = REPO_ROOT / "data" / "retrieval" / "chunks.json"
EVAL_TEST_DIR = REPO_ROOT / "tests" / "eval"

# Frozen contract (criterion 5): exactly these counts.
FROZEN_DEV_COUNT = 12
FROZEN_EVAL_COUNT = 12
FROZEN_TOTAL = 24

ALLOWED_CATEGORIES = {"AR→AR", "EN→EN", "mixed", "lookup", "factoid", "explanation"}
CROSS_DIRECTION_RE = re.compile(r"(AR\s*→\s*EN|EN\s*→\s*AR|AR\s*->\s*EN|EN\s*->\s*AR)")

IN_SCOPE_DOCS = {"DEV-001", "DEV-002", "DEV-003", "DEV-004",
                 "DEV-005", "DEV-006", "DEV-007", "DEV-010"}
EN_DOCS = {"DEV-001", "DEV-006", "DEV-010"}
AR_DOCS = {"DEV-002", "DEV-004"}
MIXED_DOCS = {"DEV-003", "DEV-005", "DEV-007"}

REQUIRED_QUESTION_KEYS = {"qid", "text", "lang", "category", "doc",
                          "targets", "split", "basis"}
OPTIONAL_QUESTION_KEYS = {"note"}

# Module-name fragments that would indicate retriever machinery.
RETRIEVER_IMPORT_MARKERS = ("retriev", "bm25", "embed", "hybrid",
                            "qdrant", "rerank", "sentence_transformer")
# Tokens that must never appear in the gold JSON (retrieval-blindness proof).
GOLD_FORBIDDEN_WORDS = ("bm25", "hybrid", "embedding", "embeddings",
                        "rrf", "rerank", "reranker", "qdrant")
GOLD_FORBIDDEN_SUBSTRINGS = ("src.retrieval", "src/retrieval",
                             "sentence_transformers", "rank_bm25",
                             "paraphrase-multilingual", "paraphrase_multilingual")


def _load_dev_file():
    raw = GOLD_PATH.read_bytes().decode("utf-8")  # strict: proves valid UTF-8
    return json.loads(raw)


def _load_eval_file():
    raw = EVAL_PATH.read_bytes().decode("utf-8")  # strict: proves valid UTF-8
    return json.loads(raw)


def _load_gold():
    """Union view (dev header + all 24 questions) preserving the R5 contract.

    Held-out exclusivity is structural: tuning paths read the dev file,
    scoring reads the eval file; the union exists only for contract checks.
    """
    dev = _load_dev_file()
    ev = _load_eval_file()
    return {"_header": dev["_header"], "questions": dev["questions"] + ev["questions"]}


def _load_dev_questions():
    """Test-only mirror of the development-tooling loader path: dev file only."""
    return _load_dev_file()["questions"]


def _load_eval_questions():
    """Test-only mirror of the scoring loader path: eval file only."""
    return _load_eval_file()["questions"]


def _load_chunk_ids():
    chunks = json.loads(CHUNKS_PATH.read_bytes().decode("utf-8"))
    return {c["chunk_id"] for c in chunks}


def test_gold_file_lives_outside_corpus():
    """Gold file must sit evaluation-side, never inside the corpus."""
    assert GOLD_PATH.is_file(), f"gold file missing: {GOLD_PATH}"
    assert GOLD_PATH.name == "gold_questions.json"
    assert GOLD_PATH.parent.name == "eval"
    assert "data" in GOLD_PATH.parts
    assert "retrieval" not in GOLD_PATH.parts, "gold must not live under data/retrieval"
    assert "documents" not in GOLD_PATH.parts, "gold must not live under the documents corpus"
    assert CHUNKS_PATH.is_file(), "chunk inventory must exist for target resolution"


def test_gold_is_valid_utf8_json():
    raw = GOLD_PATH.read_bytes()
    text = raw.decode("utf-8")  # raises on invalid UTF-8
    doc = json.loads(text)
    assert isinstance(doc, dict)
    assert set(doc.keys()) == {"_header", "questions"}


def test_header_documents_contract():
    gold = _load_gold()
    header = gold["_header"]
    for key in ("retrieval_blindness", "split_rule", "grading_rule",
                "cross_direction_backlog", "human_check_checklist",
                "language_rule", "exclusions"):
        assert key in header and header[key], f"header missing: {key}"
    assert "RETRIEVAL-BLIND" in header["retrieval_blindness"]
    assert "HELD-OUT" in header["split_rule"] or "held" in header["split_rule"].lower()
    assert isinstance(header["human_check_checklist"], list)
    assert len(header["human_check_checklist"]) >= 3
    assert header["frozen_counts"] == {"total": 24, "dev": 12, "eval": 12}


def test_schema_validity():
    gold = _load_gold()
    assert isinstance(gold["questions"], list)
    for q in gold["questions"]:
        assert REQUIRED_QUESTION_KEYS <= set(q.keys()), f"missing keys in {q.get('qid')}"
        assert set(q.keys()) <= REQUIRED_QUESTION_KEYS | OPTIONAL_QUESTION_KEYS, \
            f"unexpected keys in {q.get('qid')}"
        assert isinstance(q["text"], str) and q["text"].strip()
        assert isinstance(q["basis"], str) and q["basis"].strip()
        assert q["lang"] in {"ar", "en"}, f"bad lang in {q['qid']}"
        assert q["split"] in {"dev", "eval"}, f"bad split in {q['qid']}"
        assert isinstance(q["targets"], list) and q["targets"]
        for t in q["targets"]:
            assert set(t.keys()) == {"chunk_id", "grade"}, f"bad target in {q['qid']}"
            assert isinstance(t["chunk_id"], str) and t["chunk_id"].startswith("CHK-")


def test_qids_unique_and_prefix_matches_split():
    gold = _load_gold()
    qids = [q["qid"] for q in gold["questions"]]
    assert len(qids) == len(set(qids)), "duplicate qids"
    for q in gold["questions"]:
        if q["split"] == "dev":
            assert q["qid"].startswith("Q-DEV-"), f"dev qid prefix: {q['qid']}"
        else:
            assert q["qid"].startswith("Q-EVAL-"), f"eval qid prefix: {q['qid']}"


def test_frozen_split_counts():
    gold = _load_gold()
    dev = [q for q in gold["questions"] if q["split"] == "dev"]
    ev = [q for q in gold["questions"] if q["split"] == "eval"]
    assert len(gold["questions"]) == FROZEN_TOTAL
    assert len(dev) == FROZEN_DEV_COUNT
    assert len(ev) == FROZEN_EVAL_COUNT


def test_dev_eval_id_disjointness():
    gold = _load_gold()
    dev_ids = {q["qid"] for q in gold["questions"] if q["split"] == "dev"}
    eval_ids = {q["qid"] for q in gold["questions"] if q["split"] == "eval"}
    assert dev_ids.isdisjoint(eval_ids)


def test_all_chunk_ids_resolve_in_inventory():
    gold = _load_gold()
    chunk_ids = _load_chunk_ids()
    assert chunk_ids, "empty chunk inventory"
    for q in gold["questions"]:
        for t in q["targets"]:
            assert t["chunk_id"] in chunk_ids, \
                f"{q['qid']} references unknown chunk {t['chunk_id']}"


def test_grades_valid_with_primary_present():
    gold = _load_gold()
    for q in gold["questions"]:
        grades = [t["grade"] for t in q["targets"]]
        assert grades, f"no targets in {q['qid']}"
        assert all(g in {"primary", "acceptable"} for g in grades), \
            f"bad grade in {q['qid']}"
        assert "primary" in grades, f"no primary target in {q['qid']}"


def test_no_cross_direction_categories():
    gold = _load_gold()
    for q in gold["questions"]:
        assert q["category"] in ALLOWED_CATEGORIES, \
            f"unexpected category in {q['qid']}: {q['category']}"
        assert CROSS_DIRECTION_RE.search(q["category"]) is None, \
            f"cross-direction category in {q['qid']}"


def test_docs_all_exist_in_corpus_scope():
    gold = _load_gold()
    docs = {q["doc"] for q in gold["questions"]}
    assert docs <= IN_SCOPE_DOCS, f"out-of-scope docs: {docs - IN_SCOPE_DOCS}"
    assert "DEV-008" not in docs and "DEV-009" not in docs
    for doc in sorted(IN_SCOPE_DOCS):
        assert any(q["doc"] == doc for q in gold["questions"]), \
            f"no question for text-bearing doc {doc}"


def test_language_convention_per_doc():
    gold = _load_gold()
    for q in gold["questions"]:
        if q["doc"] in EN_DOCS:
            assert q["lang"] == "en", f"{q['qid']}: EN doc needs EN query"
        elif q["doc"] in AR_DOCS:
            assert q["lang"] == "ar", f"{q['qid']}: AR doc needs AR query"
        elif q["doc"] in MIXED_DOCS:
            assert q["lang"] == "en", f"{q['qid']}: mixed doc needs EN query"


def test_gold_json_references_no_retriever():
    """Gold file must not reference retriever modules or execution."""
    text = GOLD_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for word in GOLD_FORBIDDEN_WORDS:
        assert re.search(r"\b" + re.escape(word) + r"\b", lowered) is None, \
            f"gold JSON mentions forbidden token: {word}"
    for sub in GOLD_FORBIDDEN_SUBSTRINGS:
        assert sub not in lowered, \
            f"gold JSON mentions forbidden module ref: {sub}"


def test_eval_test_files_import_no_retriever_modules():
    """AST check: no test file in tests/eval imports retrieval machinery,
    and this integrity test itself stays stdlib-only."""
    py_files = sorted(EVAL_TEST_DIR.glob("*.py"))
    assert py_files, "no test files found in tests/eval"
    for py in py_files:
        tree = ast.parse(py.read_bytes().decode("utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for mod in modules:
                assert not any(m in mod.lower() for m in RETRIEVER_IMPORT_MARKERS), \
                    f"{py.name} imports retriever-adjacent module: {mod}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in {"__import__"}:
                raise AssertionError(f"{py.name} uses dynamic __import__")


# ---- Additive recovery tests (two-file split, provenance, sign-off) ----

def _assert_questions_schema(questions):
    assert isinstance(questions, list) and questions
    for q in questions:
        assert REQUIRED_QUESTION_KEYS <= set(q.keys()), f"missing keys in {q.get('qid')}"
        assert set(q.keys()) <= REQUIRED_QUESTION_KEYS | OPTIONAL_QUESTION_KEYS, \
            f"unexpected keys in {q.get('qid')}"
        assert isinstance(q["text"], str) and q["text"].strip()
        assert isinstance(q["basis"], str) and q["basis"].strip()
        assert q["lang"] in {"ar", "en"}, f"bad lang in {q['qid']}"
        assert q["split"] in {"dev", "eval"}, f"bad split in {q['qid']}"
        assert isinstance(q["targets"], list) and q["targets"]
        for t in q["targets"]:
            assert set(t.keys()) == {"chunk_id", "grade"}, f"bad target in {q['qid']}"
            assert isinstance(t["chunk_id"], str) and t["chunk_id"].startswith("CHK-")


def _assert_questions_resolve_and_grade(questions):
    chunks = json.loads(CHUNKS_PATH.read_bytes().decode("utf-8"))
    chunk_ids = {c["chunk_id"] for c in chunks}
    assert chunk_ids, "empty chunk inventory"
    for q in questions:
        grades = [t["grade"] for t in q["targets"]]
        assert grades, f"no targets in {q['qid']}"
        assert all(g in {"primary", "acceptable"} for g in grades), \
            f"bad grade in {q['qid']}"
        assert "primary" in grades, f"no primary target in {q['qid']}"
        for t in q["targets"]:
            assert t["chunk_id"] in chunk_ids, \
                f"{q['qid']} references unknown chunk {t['chunk_id']}"


def _assert_no_retriever_tokens(text, origin):
    lowered = text.lower()
    for word in GOLD_FORBIDDEN_WORDS:
        assert re.search(r"\b" + re.escape(word) + r"\b", lowered) is None, \
            f"{origin} mentions forbidden token: {word}"
    for sub in GOLD_FORBIDDEN_SUBSTRINGS:
        assert sub not in lowered, \
            f"{origin} mentions forbidden module ref: {sub}"


def test_eval_file_lives_outside_corpus():
    assert EVAL_PATH.is_file(), f"eval file missing: {EVAL_PATH}"
    assert EVAL_PATH.name == "gold_eval_questions.json"
    assert EVAL_PATH.parent.name == "eval"
    assert "retrieval" not in EVAL_PATH.parts
    assert "documents" not in EVAL_PATH.parts


def test_both_files_are_valid_utf8_json_with_shared_shape():
    for path in (GOLD_PATH, EVAL_PATH):
        doc = json.loads(path.read_bytes().decode("utf-8"))
        assert set(doc.keys()) == {"_header", "questions"}, f"bad top-level keys in {path.name}"
    dev = _load_dev_file()
    ev = _load_eval_file()
    _assert_questions_schema(dev["questions"])
    _assert_questions_schema(ev["questions"])


def test_dev_file_holds_only_qdev():
    dev_ids = [q["qid"] for q in _load_dev_questions()]
    assert len(dev_ids) == FROZEN_DEV_COUNT
    assert all(qid.startswith("Q-DEV-") for qid in dev_ids)
    assert all(q["split"] == "dev" for q in _load_dev_questions())


def test_eval_file_holds_only_qeval():
    eval_ids = [q["qid"] for q in _load_eval_questions()]
    assert len(eval_ids) == FROZEN_EVAL_COUNT
    assert all(qid.startswith("Q-EVAL-") for qid in eval_ids)
    assert all(q["split"] == "eval" for q in _load_eval_questions())


def test_no_question_id_in_both_files():
    dev_ids = {q["qid"] for q in _load_dev_questions()}
    eval_ids = {q["qid"] for q in _load_eval_questions()}
    assert dev_ids.isdisjoint(eval_ids), f"leaked IDs: {dev_ids & eval_ids}"


def test_split_counts_preserved_across_files():
    dev = _load_dev_questions()
    ev = _load_eval_questions()
    assert len(dev) == FROZEN_DEV_COUNT
    assert len(ev) == FROZEN_EVAL_COUNT
    assert len(dev) + len(ev) == FROZEN_TOTAL


def test_dev_loader_path_sees_no_held_out_ids():
    """Fails if any Q-EVAL id appears in the dev-loader view and vice versa."""
    dev_ids = {q["qid"] for q in _load_dev_questions()}
    eval_ids = {q["qid"] for q in _load_eval_questions()}
    assert not any(qid.startswith("Q-EVAL-") for qid in dev_ids), "held-out ID in dev file"
    assert not any(qid.startswith("Q-DEV-") for qid in eval_ids), "dev ID in eval file"


def test_both_files_chunk_resolution_and_grading():
    _assert_questions_resolve_and_grade(_load_dev_questions())
    _assert_questions_resolve_and_grade(_load_eval_questions())


def test_provenance_attestation_in_both_headers():
    for path, loader in (("dev", _load_dev_file), ("eval", _load_eval_file)):
        header = loader()["_header"]
        assert "attestation" in header and isinstance(header["attestation"], dict), \
            f"missing attestation in {path} header"
        att = header["attestation"]
        for key in ("author", "reviewer", "date", "fixtures_used",
                    "chunk_inventory_manifest_sha256", "retrieval_runs"):
            assert key in att and att[key], f"attestation missing/non-empty {key} in {path}"
        assert isinstance(att["fixtures_used"], list) and len(att["fixtures_used"]) >= 3
        assert "none" in att["retrieval_runs"].lower(), f"retrieval_runs must state none in {path}"


def _assert_valid_signoff_state(qid, rec):
    """Accept exactly the two valid per-item states; reject anything else."""
    assert set(rec.keys()) == {"verified", "verified_by", "verified_date"}, \
        f"{qid}: bad verification record keys"
    if rec["verified"] is False:
        assert rec["verified_by"] is None and rec["verified_date"] is None, \
            f"{qid}: unverified item must have null signer/date"
    elif rec["verified"] is True:
        assert isinstance(rec["verified_by"], str) and rec["verified_by"].strip(), \
            f"{qid}: verified item needs a non-empty signer"
        assert isinstance(rec["verified_date"], str) and rec["verified_date"].strip(), \
            f"{qid}: verified item needs a non-empty date"
    else:
        raise AssertionError(f"{qid}: verified must be true or false")


def test_verification_placeholders_cover_each_file():
    for name, doc in (("dev", _load_dev_file()), ("eval", _load_eval_file())):
        header = doc["_header"]
        assert "verification" in header and isinstance(header["verification"], dict)
        qids = {q["qid"] for q in doc["questions"]}
        assert set(header["verification"].keys()) == qids, \
            "verification map must cover exactly this file's questions"
        for qid, rec in header["verification"].items():
            _assert_valid_signoff_state(qid, rec)
        assert "r6_integration_note" in header and "verified==false" in header["r6_integration_note"], \
            "R6 exclusion instruction must be documented in-header"


def test_pre_gate_snapshot_all_items_unverified():
    """PRE-GATE SNAPSHOT: documents the current pre-gate state (all 24 items
    unverified). Expected to FAIL after the human gate signs items off
    in-place — that failure is the sign-off taking effect, not a regression;
    the state assertion above already accepts the post-gate state."""
    for name, doc in (("dev", _load_dev_file()), ("eval", _load_eval_file())):
        for qid, rec in doc["_header"]["verification"].items():
            assert rec["verified"] is False, \
                f"pre-gate snapshot: {qid} no longer unverified (gate sign-off in effect?)"
            assert rec["verified_by"] is None and rec["verified_date"] is None, \
                f"pre-gate snapshot: {qid} signer/date no longer null"


def test_target_document_matches_question_doc():
    chunks = {c["chunk_id"]: c for c in
              json.loads(CHUNKS_PATH.read_bytes().decode("utf-8"))}
    for q in _load_dev_questions() + _load_eval_questions():
        for t in q["targets"]:
            assert chunks[t["chunk_id"]]["document_id"] == q["doc"], \
                f"{q['qid']}: chunk {t['chunk_id']} not from {q['doc']}"


def test_eval_file_references_no_retriever():
    _assert_no_retriever_tokens(EVAL_PATH.read_text(encoding="utf-8"), "eval gold file")

def test_answer_bearing_targets_are_primary_not_acceptable():
    """Ensure answer-bearing chunks in DEV-001/002/003 are graded primary.

    Per Sol review R5 MAJOR-1: acceptable chunks must not answer alone.
    The audited answer-bearing chunks for Q-DEV-001/003/004/005/006 and
    Q-EVAL-001/002/003 must be graded 'primary', and no answer-bearing
    chunk may remain classified as 'acceptable'.
    """
    dev_qs = {q["qid"]: q for q in _load_dev_questions()}
    eval_qs = {q["qid"]: q for q in _load_eval_questions()}

    audited_qids = [
        "Q-DEV-001", "Q-DEV-003", "Q-DEV-004", "Q-DEV-005", "Q-DEV-006",
        "Q-EVAL-001", "Q-EVAL-002", "Q-EVAL-003",
    ]
    for qid in audited_qids:
        q = dev_qs.get(qid) or eval_qs.get(qid)
        assert q is not None, f"missing question {qid}"
        for t in q["targets"]:
            assert t["grade"] == "primary", (
                f"{qid} target {t['chunk_id']} has grade {t['grade']!r}, "
                f"expected 'primary' (answer-bearing chunks must not be 'acceptable')"
            )

    for q in list(dev_qs.values()) + list(eval_qs.values()):
        acc_targets = [t for t in q["targets"] if t["grade"] == "acceptable"]
        assert len(acc_targets) == 0, (
            f"{q['qid']} has acceptable targets {acc_targets}; "
            f"acceptable chunks must be genuinely supporting-only and free of answer-giving text"
        )

