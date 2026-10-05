"""Phase-4 R3 tests: multilingual semantic retriever boundary.

Per docs/RETRIEVAL_DESIGN.md sections 6 and 9. Synthetic micro-corpus
only (never ground_truth/ or canonical/ content). The provisional HF
model downloads from the Hub on first run.
"""

import ast
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from retrieval import embeddings as E  # noqa: E402

HIT_KEYS = {"chunk_id", "score", "rank", "source"}


@pytest.fixture(scope="module")
def model():
    return E.load_model()


# ---------------------------------------------------------------------------
# Identity: provisional constant + documented batching
# ---------------------------------------------------------------------------
def test_model_id_provisional_constant():
    assert E.MODEL_ID == "paraphrase-multilingual-MiniLM-L12-v2"
    doc = (E.__doc__ or "").lower()
    assert "provisional" in doc
    assert "not a hard lock" in doc


def test_batch_size_documented_constant():
    assert isinstance(E.BATCH_SIZE, int) and not isinstance(E.BATCH_SIZE, bool)
    assert 8 <= E.BATCH_SIZE <= 64
    assert E.DEVICE == "cpu"


def test_resolve_revision_recorded_nonempty():
    rev = E.resolve_revision()
    assert isinstance(rev, str) and rev
    assert re.fullmatch(r"[0-9a-f]{40}", rev), rev


def test_dependency_versions_recorded():
    versions = E.get_dependency_versions()
    for dist in ("sentence-transformers", "torch", "transformers", "numpy"):
        assert versions.get(dist), dist
        assert versions[dist] != "unknown", dist


# ---------------------------------------------------------------------------
# Encoding: determinism + L2 norms + guards
# ---------------------------------------------------------------------------
def test_encode_determinism(model):
    texts = ["The Jeddah branch reopened after refit works.",
             "مرحبا بالعالم في فرع جدة"]
    first = E.encode_texts(texts, model=model)
    second = E.encode_texts(texts, model=model)
    assert first == second


def test_encode_l2_norms(model):
    texts = ["Revenue rose in fiscal year 2023.", "صافي الدخل ارتفع"]
    vecs = E.encode_texts(texts, model=model)
    assert len(vecs) == 2
    dims = {len(v) for v in vecs}
    assert len(dims) == 1 and next(iter(dims)) > 0
    for vec in vecs:
        assert all(isinstance(x, float) for x in vec)
        assert math.isclose(math.sqrt(sum(x * x for x in vec)),
                            1.0, rel_tol=0, abs_tol=1e-5)


def test_encode_empty_list(model):
    assert E.encode_texts([], model=model) == []


def test_encode_malformed_guards(model):
    for bad in (None, "a string", 42, {"t": "x"}):
        with pytest.raises(TypeError):
            E.encode_texts(bad, model=model)
    for bad_items in ([None], [123], ["ok", None], [b"bytes"]):
        with pytest.raises(TypeError):
            E.encode_texts(bad_items, model=model)
    for bad_bs in (0, -1, "32", 2.5, True):
        with pytest.raises(TypeError):
            E.encode_texts(["hi"], model=model, batch_size=bad_bs)


# ---------------------------------------------------------------------------
# build_vectors: verbatim contract + guards
# ---------------------------------------------------------------------------
def test_build_vectors_contract(model):
    records = [
        {"chunk_id": "CHK-vec-a", "text": "The cat sits on the mat."},
        {"chunk_id": "CHK-vec-b",
         "text": "Gauge fields mediate the strong interaction."},
    ]
    vectors = E.build_vectors(records, model=model)
    assert set(vectors) == {"CHK-vec-a", "CHK-vec-b"}
    dims = {len(v) for v in vectors.values()}
    assert len(dims) == 1
    for vec in vectors.values():
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)
        assert math.isclose(math.sqrt(sum(x * x for x in vec)),
                            1.0, rel_tol=0, abs_tol=1e-5)


def test_build_vectors_empty():
    assert E.build_vectors([]) == {}


def test_build_vectors_malformed_guards():
    with pytest.raises(TypeError):
        E.build_vectors("not-a-list")
    with pytest.raises(TypeError):
        E.build_vectors([None])
    with pytest.raises(TypeError):
        E.build_vectors([{"chunk_id": "a"}])  # missing text
    with pytest.raises(TypeError):
        E.build_vectors([{"chunk_id": "a", "text": 123}])
    with pytest.raises(TypeError):
        E.build_vectors([{"chunk_id": 7, "text": "x"}])


def test_verbatim_text_usage(model, monkeypatch):
    captured = {}
    real = E.encode_texts

    def spy(texts, model=None, batch_size=None):
        captured["texts"] = list(texts)
        kwargs = {} if batch_size is None else {"batch_size": batch_size}
        return real(texts, model=model, **kwargs)

    monkeypatch.setattr(E, "encode_texts", spy)
    records = [
        {"chunk_id": "CHK-ws-1", "text": "  keep  SPACES  "},
        {"chunk_id": "CHK-ws-2", "text": "keep SPACES"},
    ]
    E.build_vectors(records, model=model)
    assert captured["texts"] == ["  keep  SPACES  ", "keep SPACES"]


# ---------------------------------------------------------------------------
# retrieve: semantic sanity + tie-break + return contract + guards
# ---------------------------------------------------------------------------
def _micro_corpus():
    return [
        {"chunk_id": "CHK-obvious1",
         "text": ("The Jeddah branch was closed for refit works in "
                  "September and October 2023, and revenue recovered "
                  "after reopening.")},
        {"chunk_id": "CHK-noise9",
         "text": ("Quantum chromodynamics describes the strong "
                  "interaction between quarks and gluons with "
                  "non-abelian gauge symmetry.")},
    ]


def test_retrieve_semantic_sanity(model):
    records = _micro_corpus()
    vectors = E.build_vectors(records, model=model)
    by_id = {r["chunk_id"]: r for r in records}
    hits = E.retrieve(
        "Jeddah store closure for renovation in September October 2023",
        2, vectors, by_id, model=model)
    assert [h["chunk_id"] for h in hits] == ["CHK-obvious1", "CHK-noise9"]
    assert hits[0]["score"] > hits[1]["score"]


def test_retrieve_tie_break(model):
    records = [
        {"chunk_id": "CHK-zzz-tie",
         "text": "identical tie text for determinism check"},
        {"chunk_id": "CHK-aaa-tie",
         "text": "identical tie text for determinism check"},
    ]
    vectors = E.build_vectors(records, model=model)
    by_id = {r["chunk_id"]: r for r in records}
    hits = E.retrieve("identical tie text query", 2, vectors, by_id,
                      model=model)
    assert [h["chunk_id"] for h in hits] == ["CHK-aaa-tie", "CHK-zzz-tie"]
    assert hits[0]["score"] == hits[1]["score"]
    assert [h["rank"] for h in hits] == [1, 2]


def test_retrieve_return_contract(model):
    records = _micro_corpus()
    vectors = E.build_vectors(records, model=model)
    by_id = {r["chunk_id"]: r for r in records}
    hits = E.retrieve("Jeddah branch refit closure", 2, vectors, by_id,
                      model=model)
    assert len(hits) == 2
    for i, hit in enumerate(hits, start=1):
        assert set(hit) == HIT_KEYS
        assert isinstance(hit["chunk_id"], str)
        assert isinstance(hit["score"], float)
        assert hit["rank"] == i
        assert hit["source"] == "semantic"
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    # top_k truncation + 1-based rank
    one = E.retrieve("Jeddah branch refit closure", 1, vectors, by_id,
                     model=model)
    assert len(one) == 1 and one[0]["rank"] == 1
    # determinism
    again = E.retrieve("Jeddah branch refit closure", 2, vectors, by_id,
                       model=model)
    assert hits == again


def test_retrieve_empty_blank(model):
    records = _micro_corpus()
    vectors = E.build_vectors(records, model=model)
    by_id = {r["chunk_id"]: r for r in records}
    for query in ("", "   ", "\t\n "):
        assert E.retrieve(query, 5, vectors, by_id, model=model) == []
    assert E.retrieve("hello", 5, {}, {}, model=model) == []


def test_retrieve_malformed_guards(model):
    records = _micro_corpus()
    vectors = E.build_vectors(records, model=model)
    by_id = {r["chunk_id"]: r for r in records}
    for bad_query in (None, 123, ["x"], {"q": 1}):
        with pytest.raises(TypeError):
            E.retrieve(bad_query, 5, vectors, by_id, model=model)
    for bad in (None, "x", [1], {"a": 1}):
        with pytest.raises(TypeError):
            E.retrieve("hi", 5, bad, by_id, model=model)
        with pytest.raises(TypeError):
            E.retrieve("hi", 5, vectors, bad, model=model)


@pytest.mark.parametrize("bad_topk",
                         [0, -1, 101, 1000, "5", 2.5, None, True, False])
def test_retrieve_topk_guards(model, bad_topk):
    records = _micro_corpus()
    vectors = E.build_vectors(records, model=model)
    by_id = {r["chunk_id"]: r for r in records}
    with pytest.raises(TypeError):
        E.retrieve("hello", bad_topk, vectors, by_id, model=model)


# ---------------------------------------------------------------------------
# Failure surface: explicit errors naming model + cache
# ---------------------------------------------------------------------------
def test_load_failure_names_model_and_cache(monkeypatch):
    import sentence_transformers
    bogus = "no-such-model-xyz-123"

    def boom(*args, **kwargs):
        raise OSError("offline simulated")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", boom)
    with pytest.raises(RuntimeError) as exc:
        E.load_model(bogus)
    message = str(exc.value).lower()
    assert bogus in str(exc.value)
    assert "cache" in message


def test_revision_failure_names_model_and_cache(monkeypatch):
    from huggingface_hub import HfApi
    bogus = "no-such-model-xyz-123"

    def boom(self, repo_id, **kwargs):
        raise OSError("offline simulated")

    monkeypatch.setattr(HfApi, "model_info", boom)
    with pytest.raises(RuntimeError) as exc:
        E.resolve_revision(bogus)
    assert bogus in str(exc.value)
    assert "cache" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Surface hygiene: no weight-update APIs referenced
# ---------------------------------------------------------------------------
def test_no_fine_tune_surface():
    forbidden = {"fit", "train", "optimizer", "optimiser", "finetune",
                 "fine_tune"}
    targets = [ROOT / "src" / "retrieval" / "embeddings.py",
               ROOT / "scripts" / "build_vectors.py"]
    for target in targets:
        tree = ast.parse(target.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                names.add(node.attr.lower())
        hit = names & forbidden
        assert not hit, "%s references %s" % (target.name, hit)


# ---------------------------------------------------------------------------
# Recovery (additive): revision pinning, eval enforcement, CLI guards,
# validation order, dependency provenance
# ---------------------------------------------------------------------------
def _load_build_vectors_module():
    import importlib.util
    path = ROOT / "scripts" / "build_vectors.py"
    spec = importlib.util.spec_from_file_location(
        "build_vectors_recovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_revision_propagates_to_loader(monkeypatch):
    import sentence_transformers
    captured = {}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def eval(self):
            return None

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer",
                        FakeModel)
    rev = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    E.load_model("no-such-model-r3-xyz", revision=rev)
    assert captured.get("revision") == rev
    assert captured.get("device") == "cpu"
    assert captured.get("trust_remote_code") is False
    captured.clear()
    E.load_model("no-such-model-r3-xyz", revision=None)
    assert "revision" in captured and captured["revision"] is None


def test_build_vectors_passes_revision(monkeypatch):
    captured = {}

    class StubModel:
        def eval(self):
            return None

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 4))

    def fake_load(model_id=None, revision=None, cache=True):
        captured["revision"] = revision
        return StubModel()

    monkeypatch.setattr(E, "load_model", fake_load)
    out = E.build_vectors([{"chunk_id": "CHK-rev-1", "text": "hello"}],
                          revision="a" * 40)
    assert captured["revision"] == "a" * 40
    assert set(out) == {"CHK-rev-1"}


def test_eval_failure_raises(monkeypatch):
    import sentence_transformers

    class BadEval:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self):
            raise RuntimeError("eval boom")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer",
                        BadEval)
    with pytest.raises(RuntimeError) as exc:
        E.load_model("no-such-model-r3-xyz", revision="b" * 40)
    assert "eval" in str(exc.value).lower()

    class BadEncode:
        def eval(self):
            raise RuntimeError("encode-eval boom")

    with pytest.raises(RuntimeError):
        E.encode_texts(["hi"], model=BadEncode())


def test_two_loaded_instance_determinism():
    first = E.load_model(cache=False)
    second = E.load_model(cache=False)
    assert first is not second
    texts = ["The Jeddah branch reopened after refit works.",
             "صافي الدخل"]
    assert (E.encode_texts(texts, model=first)
            == E.encode_texts(texts, model=second))


def test_determinism_contract_documented():
    doc = (E.__doc__ or "").lower()
    assert "determinism" in doc
    assert "section 8" in doc
    assert "not guaranteed" not in doc


def test_two_build_vector_determinism(model):
    """Authoritative two-build vector determinism contract (design section 8).

    Two independent builds on identical chunk records yield bit-identical
    vector dictionaries and serialized JSON.
    """
    records = [
        {"chunk_id": "CHK-det-01", "text": "The Jeddah branch reopened after refit works."},
        {"chunk_id": "CHK-det-02", "text": "صافي الدخل ارتفع في عام 2023."},
        {"chunk_id": "CHK-det-03", "text": "Invoice total due: 82,209.85 SAR"},
    ]
    model_b = E.load_model(cache=False)
    vecs_a = E.build_vectors(records, model=model)
    vecs_b = E.build_vectors(records, model=model_b)
    assert vecs_a == vecs_b
    assert json.dumps(vecs_a, ensure_ascii=False) == json.dumps(vecs_b, ensure_ascii=False)


def test_build_vectors_updates_index_meta(tmp_path):
    """build_vectors CLI updates index_meta.json in-place per design section 8."""
    bv = _load_build_vectors_module()
    out = tmp_path / "out"
    out.mkdir()
    index_meta_path = out / "index_meta.json"
    initial_meta = {
        "chunker": {
            "name": "mizaniq-text-chunker",
            "version": "1.0.0",
            "params": {"chunk_id": "CHK-hash"},
        },
        "embedding_model": {
            "id": "pending",
            "revision": "pending",
            "note": "R1 text-only",
        },
        "corpus": {"manifest_sha256": "fake-hash-123"},
        "counts": {"chunks_total": 5, "by_kind": {"text": 5}},
        "timings_ms": {"chunk": 25.0},
    }
    index_meta_path.write_text(
        json.dumps(initial_meta, indent=2) + "\n", encoding="utf-8")

    updated = bv.update_index_meta(
        index_meta_path,
        model_id=E.MODEL_ID,
        repo="sentence-transformers/" + E.MODEL_ID,
        revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        dims=384,
        vector_count=5,
        encode_ms=120.0,
    )
    assert updated is True
    meta = json.loads(index_meta_path.read_text(encoding="utf-8"))
    # R1 fields preserved
    assert meta["chunker"]["name"] == "mizaniq-text-chunker"
    assert meta["corpus"]["manifest_sha256"] == "fake-hash-123"
    assert meta["counts"]["chunks_total"] == 5
    assert meta["timings_ms"]["chunk"] == 25.0
    # Embedding model recorded per section 8
    assert meta["embedding_model"]["id"] == E.MODEL_ID
    assert meta["embedding_model"]["revision"] == "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    assert meta["embedding_model"]["revision_pinned"] is True
    assert meta["counts"]["vectors_total"] == 5
    assert meta["counts"]["dims"] == 384
    assert meta["timings_ms"]["vector_encode"] == 120.0


def test_real_index_meta_records_model_id_and_revision():
    """Verify data/retrieval/index_meta.json records model id and revision."""
    index_meta_path = ROOT / "data" / "retrieval" / "index_meta.json"
    assert index_meta_path.is_file()
    meta = json.loads(index_meta_path.read_text(encoding="utf-8"))
    emb = meta.get("embedding_model", {})
    assert emb.get("id") == E.MODEL_ID
    assert emb.get("id") != "pending"
    assert emb.get("revision") != "pending"
    assert re.fullmatch(r"[0-9a-f]{40}", emb.get("revision", "")), emb.get("revision")
    # Verify R1 chunker and corpus metadata are intact
    assert meta.get("chunker", {}).get("name") == "mizaniq-text-chunker"
    assert meta.get("counts", {}).get("chunks_total") == 4388
    assert meta.get("corpus", {}).get("manifest_sha256")


def test_retrieve_validates_before_blank_shortcut(model):
    records = _micro_corpus()
    vectors = E.build_vectors(records, model=model)
    by_id = {r["chunk_id"]: r for r in records}
    with pytest.raises(TypeError):
        E.retrieve("", 0, vectors, by_id, model=model)
    with pytest.raises(TypeError):
        E.retrieve("   ", "5", vectors, by_id, model=model)
    with pytest.raises(TypeError):
        E.retrieve("", 5, None, by_id, model=model)
    with pytest.raises(TypeError):
        E.retrieve("", 5, vectors, None, model=model)
    with pytest.raises(TypeError):
        E.retrieve("\t", 101, vectors, {"x": "not-a-dict"}, model=model)
    assert E.retrieve("", 5, vectors, by_id, model=model) == []


def test_cli_output_restrictions():
    bv = _load_build_vectors_module()
    out = ROOT / "data" / "retrieval"
    assert bv._resolve_output(None, "vectors.json", out).name == "vectors.json"
    assert (bv._resolve_output("custom.json", "vectors.json", out).name
            == "custom.json")
    for bad in ("chunks.json", "index_meta.json", "requirements.txt",
                "../x.json", "sub/dir.json", "/abs.json", "C:evil.json",
                "..", ""):
        with pytest.raises(ValueError):
            bv._resolve_output(bad, "vectors.json", out)


def test_cli_rejects_protected_outputs(monkeypatch, tmp_path):
    bv = _load_build_vectors_module()
    monkeypatch.setattr(
        sys, "argv",
        ["build_vectors.py", "--out-dir", str(tmp_path),
         "--vectors", "chunks.json"])
    assert bv.main() == 2
    monkeypatch.setattr(
        sys, "argv",
        ["build_vectors.py", "--out-dir", str(tmp_path),
         "--meta", "../index_meta.json"])
    assert bv.main() == 2
    monkeypatch.setattr(
        sys, "argv",
        ["build_vectors.py", "--out-dir", str(tmp_path),
         "--vectors", "same.json", "--meta", "same.json"])
    assert bv.main() == 2


def test_dependency_versions_include_scipy():
    versions = E.get_dependency_versions()
    for dist in ("scipy", "tokenizers", "huggingface_hub"):
        assert versions.get(dist) not in (None, "", "unknown"), dist


# ---------------------------------------------------------------------------
# CLI confinement follow-up (additive): case-insensitive protection,
# input/output collision, prohibited input corpora
# ---------------------------------------------------------------------------
def test_cli_case_variant_protected_rejected():
    bv = _load_build_vectors_module()
    out = ROOT / "data" / "retrieval"
    for bad in ("CHUNKS.JSON", "Chunks.Json", "cHuNkS.jSoN",
                "INDEX_META.JSON", "Index_Meta.Json",
                "REQUIREMENTS.TXT", "Requirements.Txt"):
        with pytest.raises(ValueError):
            bv._resolve_output(bad, "vectors.json", out)


def test_cli_chunks_output_collision_rejected(monkeypatch, tmp_path):
    bv = _load_build_vectors_module()
    monkeypatch.setattr(
        sys, "argv",
        ["build_vectors.py", "--out-dir", str(tmp_path),
         "--chunks", str(tmp_path / "vectors.json"),
         "--vectors", "vectors.json"])
    assert bv.main() == 2
    monkeypatch.setattr(
        sys, "argv",
        ["build_vectors.py", "--out-dir", str(tmp_path),
         "--chunks", str(tmp_path / "vectors_meta.json"),
         "--meta", "vectors_meta.json"])
    assert bv.main() == 2


def test_cli_prohibited_inputs_rejected(monkeypatch, tmp_path):
    bv = _load_build_vectors_module()
    for bad in (ROOT / "data" / "canonical" / "x.csv",
                ROOT / "data" / "eval" / "q.json",
                ROOT / "data" / "dev" / "dataset_v0.1" / "documents"):
        monkeypatch.setattr(
            sys, "argv",
            ["build_vectors.py", "--chunks", str(bad),
             "--out-dir", str(tmp_path)])
        assert bv.main() == 2
    # Helper level as well (no argv needed).
    with pytest.raises(ValueError):
        bv._resolve_chunks(
            str(ROOT / "data" / "canonical" / "x.csv"), tmp_path)
    with pytest.raises(ValueError):
        bv._resolve_chunks(str(ROOT / "data" / "eval" / "q.json"), tmp_path)


def test_cli_legitimate_inputs_accepted():
    bv = _load_build_vectors_module()
    out = ROOT / "data" / "retrieval"
    assert bv._resolve_chunks(None, out) == out / "chunks.json"
    assert (bv._resolve_chunks(str(out / "chunks.json"), out)
            == out / "chunks.json")
