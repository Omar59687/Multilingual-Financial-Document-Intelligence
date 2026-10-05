"""RRF hybrid retriever + unified retrieval API (Phase 4 R4).

Implements docs/RETRIEVAL_DESIGN.md section 7 (hybrid) and the unified
subset of section 9 (``retrieve(query, top_k, mode)`` plus
``get_chunk(chunk_id)``) over the accepted R1/R2/R3 implementation.
No new retrieval architecture is invented here: BM25 behavior is owned
by ``retrieval.bm25.BM25Index`` (R2, stdlib, frozen ``K1``/``B``) and
semantic scoring semantics are owned by ``retrieval.embeddings`` (R3,
provisional model, L2 cosine, brute force). This module only fuses
their FULL-DEPTH rankings and exposes the three-mode entry point that
Phase 5 / G2 measurement calls.

Fusion (design section 7, frozen, never tuned)::

    score(chunk) = SUM_over_sides_present(1 / (RRF_K + rank_side(chunk)))

``RRF_K = 60`` is a module constant (THE documented constant). Both
input rankings are consumed at FULL ranking depth (never top-K
truncated): the BM25 side is derived from ``BM25Index.score`` (the
complete match dict, sorted) and the semantic side is a brute-force
cosine scan over every stored vector, because the public per-mode
``retrieve`` entry points cap ``top_k`` at 100 while the corpus holds
thousands of chunks. A chunk missing from one side contributes nothing
from that side. Input per-side scores are intentionally ignored (RRF
needs no score calibration); only per-side ranks matter. Final order
is ``(rrf_score desc, chunk_id asc)`` with 1-based sequential ranks.

Unified API mapping to design section 9:

- ``UnifiedRetriever.retrieve(query, top_k=10, mode="hybrid")`` with
  ``mode in {"bm25", "semantic", "hybrid"}`` returns ranked
  ``[{chunk_id, score, rank, source}]`` where ``source`` is the mode
  name (``"bm25"`` / ``"semantic"`` / ``"hybrid"``). ``mode="bm25"``
  and ``mode="semantic"`` delegate verbatim to the accepted R2/R3
  entry points (identical ordering, scores, sources, and their
  quirks); ``mode="hybrid"`` fuses full-depth rankings per above.
- ``UnifiedRetriever.get_chunk(chunk_id)`` returns the stored chunk
  record (verbatim text plus R1 provenance/metadata).
- Ordering is deterministic everywhere: ``(score desc, chunk_id asc)``.
- ``top_k`` validated (``1 <= top_k <= 100``, else ``TypeError``;
  ``bool`` is rejected explicitly).
- Empty/blank (whitespace-only) queries return ``[]`` for every mode
  (never an error, never fabricated hits) WITHOUT loading the
  embedding model.
- Validation order is fixed: ``query`` type, ``top_k`` type/bounds,
  ``mode`` value, ``chunks_by_id`` type are all checked BEFORE the
  blank-query shortcut (mirrors the R3 ``validates-before-blank``
  contract), so malformed calls raise even for blank queries.
- Malformed inputs raise ``TypeError``; unknown ``chunk_id`` raises
  ``KeyError``; missing on-disk index artifacts (``chunks.json``,
  ``vectors.json``, and the required ``index_meta.json``) raise
  ``FileNotFoundError`` with an explicit path message (never silent).
  Malformed ``index_meta.json`` raises ``TypeError``; R1-only
  (``"pending"``) embedding metadata raises ``RuntimeError`` (stored
  vectors cannot be attributed to pinned weights).
- Latency is measured per successful call (``time.perf_counter``) and
  exposed as ``retriever.last_latency_ms`` (plus ``last_mode`` /
  ``last_top_k``) for EVAL section 11 reporting; no thresholds.
- ``chunks_by_id`` (``dict`` or ``None``) is type-checked on every
  mode before the blank-query shortcut. Entry contents (``str`` keys,
  ``dict`` values) are additionally validated on the ``semantic`` path
  before blank handling and before any model load, mirroring accepted
  R3. Ranking effect differs per mode by prior contract: ``bm25``
  restricts to the mapping keys (R2); ``hybrid`` restricts both sides
  to the keys (R4); ``semantic`` validates the mapping but scores the
  full vector store (accepted R3 validate-but-ignore behavior,
  preserved verbatim by delegation).
- Vector/chunk inventory integrity is enforced at construction: every
  vector ID must exist in the chunk inventory (orphans raise
  ``TypeError`` listing them in artifact order), so no hit can carry
  an ID that ``get_chunk`` cannot resolve. Chunks without vectors are
  allowed (they stay BM25-retrievable).
- ``from_artifacts`` records the authoritative embedding model id AND
  pinned revision from ``index_meta.json`` and uses both for every
  lazy query encoding, so query vectors always come from the recorded
  weights (design sections 6/8; the R3 no-hardcoded-hash rule is kept:
  the recorded hash is consumed, never re-pinned here).
- Returned ``chunk_id`` values always resolve via ``get_chunk``; IDs
  are never invented. Chunk records (provenance/metadata) are stored
  verbatim from the R1 chunk inventory.

R4 does NOT touch the R3 non-blocking MINOR follow-ups (swallowed
deterministic-algorithm failures; non-atomic ``index_meta.json``
update): those remain recorded in ``.factory/state/phase4-text-rag.md``
and are out of scope unless R4 genuinely depends on them (it does not:
this module calls the accepted R3 entry points as-is).
"""

import time
from pathlib import Path

import json
import re

import numpy as np

from retrieval.bm25 import BM25Index
from retrieval.bm25 import SOURCE as BM25_SOURCE
from retrieval import embeddings as emb

# Frozen RRF constant (design section 7: documented constant, never tuned).
RRF_K = 60

# Source tag stamped on fused hits (design section 9: source = mode name).
SOURCE = "hybrid"

# Modes accepted by UnifiedRetriever.retrieve (design section 9).
MODES = ("bm25", "semantic", "hybrid")

# top_k validation bounds (design section 9: 1 <= top_k <= 100).
TOP_K_MIN = 1
TOP_K_MAX = 100

__all__ = ["RRF_K", "SOURCE", "MODES", "TOP_K_MIN", "TOP_K_MAX",
           "rrf_fuse", "UnifiedRetriever",
           "load_chunks", "load_vectors", "load_index_meta"]


# Pinned-revision shape per the R3 contract (40 lowercase hex).
_REVISION_RE = re.compile(r"[0-9a-f]{40}")


def _check_top_k(top_k):
    """Validate top_k per design section 9 (1 <= top_k <= 100)."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(
            "top_k must be an int with 1 <= top_k <= 100, "
            "got %r" % (top_k,))
    if not (TOP_K_MIN <= top_k <= TOP_K_MAX):
        raise TypeError(
            "top_k must satisfy 1 <= top_k <= 100, got %r" % (top_k,))
    return top_k


def _check_k(k):
    """Validate the RRF depth constant (product path always uses RRF_K)."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be a positive int, got %r" % (k,))
    if k <= 0:
        raise TypeError("k must be a positive int, got %r" % (k,))
    return k


def _check_mapping_entries(mapping):
    """Validate chunks_by_id ENTRIES (mirrors accepted R3 ordering).

    Keys must be ``str`` and values chunk-record ``dict``s, else
    ``TypeError`` with the same messages as
    ``retrieval.embeddings.retrieve``. Callers invoke this BEFORE the
    blank-query shortcut and BEFORE any model load, restoring the R3
    validates-before-blank contract on the semantic path.
    """
    for cid, record in mapping.items():
        if not isinstance(cid, str):
            raise TypeError(
                "chunks_by_id key must be str, got %s"
                % type(cid).__name__)
        if not isinstance(record, dict):
            raise TypeError(
                "chunks_by_id[%r] must be a chunk dict, got %s"
                % (cid, type(record).__name__))
    return mapping


def _rank_map(ranked, side):
    """Map a full-depth ranked list to {chunk_id: rank}.

    Each item must be a dict holding a str ``chunk_id`` and an int
    ``rank`` >= 1 (``bool`` rejected). Duplicate chunk IDs within one
    side raise ``TypeError`` (a well-formed ranking lists each chunk
    once). Per-side scores are deliberately NOT read: RRF fuses ranks
    only (no score calibration, design section 7).
    """
    if not isinstance(ranked, list):
        raise TypeError(
            "%s ranking must be a list, got %s" % (side, type(ranked).__name__))
    mapping = {}
    for position, item in enumerate(ranked):
        if not isinstance(item, dict):
            raise TypeError(
                "%s ranking item %d must be a dict, got %s"
                % (side, position, type(item).__name__))
        cid = item.get("chunk_id")
        rank = item.get("rank")
        if not isinstance(cid, str):
            raise TypeError(
                "%s ranking item %d has non-str chunk_id: %r"
                % (side, position, cid))
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise TypeError(
                "%s ranking item %d has invalid rank: %r"
                % (side, position, rank))
        if cid in mapping:
            raise TypeError(
                "%s ranking holds duplicate chunk_id: %r" % (side, cid))
        mapping[cid] = rank
    return mapping


def rrf_fuse(bm25_ranked, semantic_ranked, top_k, k=RRF_K):
    """Fuse two FULL-DEPTH rankings with Reciprocal Rank Fusion.

    ``bm25_ranked`` / ``semantic_ranked`` are ranked lists of
    ``{chunk_id, rank, ...}`` (ranks 1-based over the full corpus, not
    top-K truncated; either side may be ``[]``). ``score(chunk_id)``
    sums ``1.0 / (k + rank)`` over the sides containing it; a chunk
    missing from one side contributes nothing from that side. Output
    is ``[{chunk_id, score, rank, source}]`` with ``source="hybrid"``,
    ordered ``(score desc, chunk_id asc)``, truncated to ``top_k``.
    Overlapping chunks appear exactly once. ``k`` defaults to the
    frozen ``RRF_K`` (the product path never overrides it).
    """
    _check_top_k(top_k)
    _check_k(k)
    bm25_ranks = _rank_map(bm25_ranked, "bm25")
    sem_ranks = _rank_map(semantic_ranked, "semantic")
    scores = {}
    for cid, rank in bm25_ranks.items():
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for cid, rank in sem_ranks.items():
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"chunk_id": cid, "score": float(value), "rank": rank,
         "source": SOURCE}
        for rank, (cid, value) in enumerate(ordered[:top_k], start=1)
    ]


def _default_data_dir():
    """Repo-root ``data/retrieval`` anchor (mirrors chunking defaults)."""
    return Path(__file__).resolve().parents[2] / "data" / "retrieval"


def load_chunks(path):
    """Load a chunk-inventory JSON list or raise explicitly.

    Missing path raises ``FileNotFoundError`` naming the artifact;
    non-list JSON raises ``TypeError`` (never silent, never invented).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "missing index artifact (chunks inventory): %s "
            "(build it with scripts/build_retrieval_index.py)" % (path,))
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(
            "chunks artifact %s must hold a JSON list, got %s"
            % (path, type(records).__name__))
    return records


def load_vectors(path):
    """Load a vectors JSON dict or raise explicitly.

    Missing path raises ``FileNotFoundError`` naming the artifact;
    non-dict JSON raises ``TypeError``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "missing index artifact (vectors store): %s "
            "(build it with scripts/build_vectors.py)" % (path,))
    vectors = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(vectors, dict):
        raise TypeError(
            "vectors artifact %s must hold a JSON object, got %s"
            % (path, type(vectors).__name__))
    return vectors


def load_index_meta(path):
    """Load and validate the authoritative index metadata artifact.

    Returns the parsed ``index_meta.json`` dict. Missing path raises
    ``FileNotFoundError`` naming the artifact (the metadata is required
    wherever the repository contract requires pinned weights: lazy
    query encoding must use the recorded model id AND revision).
    Non-dict JSON, a missing/malformed ``embedding_model`` section, or
    missing/non-str/non-conforming id/revision values raise
    ``TypeError``. R1-only ``"pending"`` id/revision values raise
    ``RuntimeError`` (incompatible: no recorded weights to pin to).
    Any recorded 40-hex revision is accepted as-is (swap path stays
    open; nothing is re-pinned here).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "missing index artifact (index metadata): %s "
            "(build it with scripts/build_retrieval_index.py and "
            "scripts/build_vectors.py)" % (path,))
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise TypeError(
            "index metadata %s must hold a JSON object, got %s"
            % (path, type(meta).__name__))
    section = meta.get("embedding_model")
    if not isinstance(section, dict):
        raise TypeError(
            "index metadata %s has a missing/malformed "
            "'embedding_model' section: %r" % (path, section))
    model_id = section.get("id")
    revision = section.get("revision")
    if not isinstance(model_id, str) or not model_id:
        raise TypeError(
            "index metadata %s has a missing/malformed "
            "embedding_model.id: %r" % (path, model_id))
    if not isinstance(revision, str) or not revision:
        raise TypeError(
            "index metadata %s has a missing/malformed "
            "embedding_model.revision: %r" % (path, revision))
    if model_id == "pending" or revision == "pending":
        raise RuntimeError(
            "index metadata %s records pending embedding weights "
            "(id=%r, revision=%r): run scripts/build_vectors.py to pin "
            "the model id and revision before semantic/hybrid use"
            % (path, model_id, revision))
    if _REVISION_RE.fullmatch(revision) is None:
        raise TypeError(
            "index metadata %s has a malformed embedding_model.revision "
            "(expected 40 lowercase hex): %r" % (path, revision))
    return meta


class UnifiedRetriever:
    """Three-mode first-stage retriever over one frozen chunk inventory.

    ``chunks`` is the R1 chunk-record list (verbatim text +
    provenance/metadata; only ``chunk_id``/``text`` are read for
    indexing, full records are kept for ``get_chunk``).     ``vectors``
    is ``{chunk_id: [float, ...]}`` (R3 store) or ``None`` when only
    ``mode="bm25"`` will ever be used; every vector ID must exist in
    ``chunks`` (orphans raise ``TypeError`` at construction, so no hit
    can carry an unresolvable ID; chunks without vectors are allowed).
    ``model`` is a loaded sentence-transformers instance or ``None``
    (lazy-loaded on the first non-blank semantic/hybrid call).
    ``model_id``/``model_revision`` pin the weights used for that lazy
    load (``from_artifacts`` always sets both from ``index_meta.json``;
    direct construction defaults to the accepted R3 loader defaults).
    Use ``from_artifacts`` to load the frozen on-disk index (missing
    files raise ``FileNotFoundError``; pending/malformed embedding
    metadata raises explicitly).
    """

    def __init__(self, chunks, vectors=None, model=None,
                 model_id=None, model_revision=None):
        if not isinstance(chunks, list):
            raise TypeError(
                "chunks must be a list of chunk records, got %s"
                % type(chunks).__name__)
        by_id = {}
        for position, record in enumerate(chunks):
            if not isinstance(record, dict):
                raise TypeError(
                    "chunk record %d must be a dict, got %s"
                    % (position, type(record).__name__))
            cid = record.get("chunk_id")
            text = record.get("text")
            if not isinstance(cid, str):
                raise TypeError(
                    "chunk record %d has non-str chunk_id: %r"
                    % (position, cid))
            if not isinstance(text, str):
                raise TypeError(
                    "chunk %r has non-str text: %r" % (cid, text))
            if cid in by_id:
                raise TypeError("duplicate chunk_id: %r" % (cid,))
            by_id[cid] = record
        if vectors is not None and not isinstance(vectors, dict):
            raise TypeError(
                "vectors must be a dict or None, got %s"
                % type(vectors).__name__)
        if vectors:
            orphans = [cid for cid in vectors if cid not in by_id]
            if orphans:
                raise TypeError(
                    "vectors hold %d chunk ID(s) absent from the chunk "
                    "inventory (unresolvable hits): %r"
                    % (len(orphans), orphans))
        if model_id is not None and not isinstance(model_id, str):
            raise TypeError(
                "model_id must be a str or None, got %s"
                % type(model_id).__name__)
        if model_revision is not None and not isinstance(model_revision, str):
            raise TypeError(
                "model_revision must be a str or None, got %s"
                % type(model_revision).__name__)
        self._chunks_by_id = by_id
        self._vectors = vectors
        self._model = model
        self._model_id = model_id
        self._model_revision = model_revision
        self._bm25 = BM25Index(chunks)
        self.last_latency_ms = None
        self.last_mode = None
        self.last_top_k = None

    @classmethod
    def from_artifacts(cls, data_dir=None, model=None):
        """Load the frozen index (chunks + vectors + required metadata).

        Missing ``chunks.json``/``vectors.json``/``index_meta.json``
        raises ``FileNotFoundError`` naming the absent artifact (never
        silent, never an empty fallback index). The recorded embedding
        model id AND pinned revision are validated via
        ``load_index_meta`` and used for every lazy query encoding, so
        query vectors always come from the recorded weights.
        """
        base = Path(data_dir) if data_dir is not None else _default_data_dir()
        chunks = load_chunks(base / "chunks.json")
        vectors = load_vectors(base / "vectors.json")
        meta = load_index_meta(base / "index_meta.json")
        section = meta["embedding_model"]
        return cls(chunks, vectors=vectors, model=model,
                   model_id=section["id"],
                   model_revision=section["revision"])

    def get_chunk(self, chunk_id):
        """Return the stored chunk record for ``chunk_id``.

        Records carry verbatim text plus R1 provenance/metadata
        (``document_id``, ``filename``, ``file_type``, ``language``,
        ``page``/``sheet``, ``element_ids``, ``kind``, ``section``,
        ``spans``, ``length``). Non-str IDs raise ``TypeError``;
        unknown IDs raise ``KeyError`` (never invented content).
        """
        if not isinstance(chunk_id, str):
            raise TypeError(
                "chunk_id must be str, got %s" % type(chunk_id).__name__)
        try:
            return self._chunks_by_id[chunk_id]
        except KeyError:
            raise KeyError("unknown chunk_id: %r" % (chunk_id,)) from None

    def _model_or_load(self):
        """Return the embedding model, lazy-loading it once if needed.

        The recorded model id AND pinned revision (when set via
        ``from_artifacts``) are passed through to the accepted R3
        loader, so recorded weights and query-encoding weights are the
        same snapshot. Direct construction without pins uses the R3
        loader defaults.
        """
        if self._model is None:
            self._model = emb.load_model(model_id=self._model_id,
                                         revision=self._model_revision)
        return self._model

    def _full_bm25_ranking(self, query, allowed):
        """Full-depth BM25 ranking (every match, never top-K truncated)."""
        scores = self._bm25.score(query)
        if allowed is not None:
            scores = {cid: value for cid, value in scores.items()
                      if cid in allowed}
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            {"chunk_id": cid, "score": value, "rank": rank,
             "source": BM25_SOURCE}
            for rank, (cid, value) in enumerate(ordered, start=1)
        ]

    def _full_semantic_ranking(self, query, allowed):
        """Full-depth brute-force cosine ranking (mirrors R3 scoring).

        Cosine = dot product of L2-normalized vectors; order is
        ``(score desc, chunk_id asc)`` exactly as in
        ``retrieval.embeddings.retrieve``. The public R3 entry point
        cannot supply full depth (its ``top_k`` caps at 100), so the
        scan is performed here at full depth per design section 7;
        scoring semantics stay identical to R3.
        """
        vectors = self._vectors
        if vectors is None:
            raise RuntimeError(
                "semantic/hybrid retrieval requires vectors "
                "(pass vectors=... or load via from_artifacts); "
                "got vectors=None")
        if not vectors:
            return []
        model = self._model_or_load()
        qvec = emb.encode_texts([query], model=model)[0]
        dim = len(qvec)
        try:
            qarr = np.asarray(qvec, dtype=np.float64)
        except Exception as exc:
            raise TypeError("query vector is not numeric: %s" % (exc,))
        ids = []
        rows = []
        for cid, vec in vectors.items():
            if allowed is not None and cid not in allowed:
                continue
            if not isinstance(cid, str):
                raise TypeError(
                    "vector key must be str, got %s" % type(cid).__name__)
            if not isinstance(vec, (list, tuple, np.ndarray)):
                raise TypeError(
                    "vector for %r must be a numeric sequence, got %s"
                    % (cid, type(vec).__name__))
            try:
                arr = np.asarray(vec, dtype=np.float64).ravel()
            except Exception as exc:
                raise TypeError(
                    "vector for %r is not numeric: %s" % (cid, exc))
            if arr.ndim != 1 or arr.shape[0] != dim:
                raise TypeError(
                    "vector for %r has dim %s, expected %d"
                    % (cid, arr.shape, dim))
            ids.append(cid)
            rows.append(arr)
        if not ids:
            return []
        scores = np.stack(rows) @ qarr
        order = sorted(range(len(ids)),
                       key=lambda i: (-float(scores[i]), ids[i]))
        return [
            {"chunk_id": ids[i], "score": float(scores[i]),
             "rank": rank, "source": emb.SOURCE_NAME}
            for rank, i in enumerate(order, start=1)
        ]

    def retrieve(self, query, top_k=10, mode="hybrid", chunks_by_id=None):
        """Rank chunks for ``query`` in the requested ``mode``.

        ``mode`` is ``"bm25"``, ``"semantic"``, or ``"hybrid"``
        (anything else raises ``TypeError``). Returns ranked
        ``[{chunk_id, score, rank, source}]`` with ``source`` equal to
        the mode name, ordered ``(score desc, chunk_id asc)`` with
        1-based sequential ranks, truncated to ``top_k``. ``bm25`` and
        ``semantic`` delegate verbatim to the accepted R2/R3 entry
        points; ``hybrid`` fuses their full-depth rankings with RRF
        (``RRF_K=60``). Empty/blank queries return ``[]``. Per-call
        latency is recorded on ``last_latency_ms``.
        """
        if not isinstance(query, str):
            raise TypeError(
                "query must be str, got %s" % type(query).__name__)
        _check_top_k(top_k)
        if not isinstance(mode, str) or mode not in MODES:
            raise TypeError(
                "mode must be one of %r, got %r" % (list(MODES), mode))
        if chunks_by_id is not None and not isinstance(chunks_by_id, dict):
            raise TypeError(
                "chunks_by_id must be a dict or None, got %s"
                % type(chunks_by_id).__name__)
        if mode == "semantic" and chunks_by_id is not None:
            # Restore accepted R3 ordering: entry validation BEFORE the
            # blank-query shortcut and BEFORE any model load.
            _check_mapping_entries(chunks_by_id)
        allowed = None if chunks_by_id is None else set(chunks_by_id)
        started = time.perf_counter()
        try:
            if not query.strip():
                return []
            if mode == "bm25":
                return self._bm25.retrieve(
                    query, top_k=top_k, chunks_by_id=chunks_by_id)
            if mode == "semantic":
                if self._vectors is None:
                    raise RuntimeError(
                        "semantic retrieval requires vectors "
                        "(pass vectors=... or load via from_artifacts); "
                        "got vectors=None")
                scope = (self._chunks_by_id if chunks_by_id is None
                         else chunks_by_id)
                if not self._vectors:
                    # Mirror R3 empty-store behavior ([] without model load).
                    return emb.retrieve(query, top_k, self._vectors, scope,
                                        model=self._model)
                return emb.retrieve(query, top_k, self._vectors, scope,
                                    model=self._model_or_load())
            full_bm25 = self._full_bm25_ranking(query, allowed)
            full_sem = self._full_semantic_ranking(query, allowed)
            return rrf_fuse(full_bm25, full_sem, top_k)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.last_latency_ms = elapsed_ms
            self.last_mode = mode
            self.last_top_k = top_k
