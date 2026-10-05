"""Multilingual semantic retriever for MizanIQ Text RAG (Phase 4 R3).

Provisional embedding boundary per docs/RETRIEVAL_DESIGN.md section 6:
``paraphrase-multilingual-MiniLM-L12-v2`` (sentence-transformers, CPU,
eval mode, L2-normalized vectors, brute-force cosine per section 9).
This identity is provisional, NOT a hard lock: the exact HF revision
commit hash is recorded at build time (see ``resolve_revision``) and
the swap path stays open (challenger models for a future
embedding-benchmark task live in the design doc, not here).

Determinism: two independent builds produce identical vectors per
docs/RETRIEVAL_DESIGN.md section 8 (excluding generated_at). Pinned for
a build: the model identity plus the resolved HF revision (passed to the
loader, so recorded hash and loaded weights are the same snapshot),
verified eval mode, CPU device, fixed batch size, verbatim text,
deterministic PyTorch algorithms, and the brute-force tie-break
(score desc, chunk_id asc). Truncation, if any, is the library
tokenizer's deterministic behavior; chunk text is never padded or
invented here.

Frozen weights only: no weight-update code paths exist in this module.
No network calls except HF Hub model/revision download at build time
(shared HF cache outside the repo, no secrets, no API keys).
"""

import os

import numpy as np

# Provisional model identity per design section 6. Module constant on
# purpose (single swap point); NOT a hard lock -- the resolved HF
# commit hash is recorded alongside every build.
MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"

# Fixed encode batch size: modest, documented, never tuned.
BATCH_SIZE = 32

# CPU-only execution (project constraint).
DEVICE = "cpu"

# Retrieval source tag for the Phase-5-facing return contract.
SOURCE_NAME = "semantic"

# top_k bounds per design section 9 (violations raise TypeError).
TOP_K_MIN = 1
TOP_K_MAX = 100

# Singleton model cache: one loaded instance per model id.
_MODEL_CACHE = {}


def _hub_cache_dir():
    """Shared HF Hub cache location (outside the repo)."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return str(HF_HUB_CACHE)
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".cache",
                            "huggingface", "hub")


def _full_repo_id(model_id=None):
    """Fully-qualified HF repo for a (possibly short) model identity."""
    mid = model_id or MODEL_ID
    if "/" in mid:
        return mid
    return "sentence-transformers/" + mid


def resolve_revision(model_id=None):
    """Record the exact HF commit hash for the embedding model.

    No hardcoded hash lives here (it would rot): the hash is resolved
    live from the Hub at build time and returned for metadata.
    Offline/unresolvable revisions raise an explicit RuntimeError naming
    the model plus the cache location (never silent, never a fallback
    to another model).
    """
    mid = model_id or MODEL_ID
    repo = _full_repo_id(mid)
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo)
        sha = getattr(info, "sha", None)
        if not sha or not isinstance(sha, str):
            raise RuntimeError("Hub returned an empty revision")
        return sha
    except Exception as exc:
        cache = _hub_cache_dir()
        raise RuntimeError(
            "cannot resolve revision for embedding model %r "
            "(HF repo %r): %s; HF cache: %s" % (mid, repo, exc, cache)
        ) from exc


def load_model(model_id=None, revision=None, cache=True):
    """Load the provisional sentence-transformers model (CPU, eval mode).

    The resolved ``revision`` (see ``resolve_revision``) is passed
    through to the loader, so the recorded hash and the loaded weights
    are the same snapshot; ``revision=None`` loads the Hub default.
    Cached per (model id, revision) unless ``cache`` is False. Eval mode
    is verified: failure to enter eval mode raises an explicit
    RuntimeError (never silently ignored). Other failures (missing
    weights, offline Hub) raise an explicit RuntimeError naming the
    model plus the cache location; this function never falls back to
    another model silently.
    """
    mid = model_id or MODEL_ID
    key = (mid, revision)
    if cache and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    hub_cache = _hub_cache_dir()
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "cannot load embedding model %r: sentence-transformers "
            "import failed (%s); HF cache: %s" % (mid, exc, hub_cache)
        ) from exc
    repo = _full_repo_id(mid)
    try:
        model = SentenceTransformer(repo, device=DEVICE,
                                    trust_remote_code=False,
                                    revision=revision)
    except Exception as exc:
        raise RuntimeError(
            "cannot load embedding model %r (HF repo %r, revision %r): "
            "%s; HF cache: %s; check network access to the HF Hub" %
            (mid, repo, revision, exc, hub_cache)
        ) from exc
    try:
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            "embedding model %r (HF repo %r, revision %r) failed to "
            "enter eval mode: %s; HF cache: %s"
            % (mid, repo, revision, exc, hub_cache)) from exc
    try:
        import torch
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if cache:
        _MODEL_CACHE[key] = model
    return model


def encode_texts(texts, model=None, batch_size=None):
    """L2-normalized sentence embeddings on CPU.

    Input: list of strings (uses verbatim text, never synthesizes).
    Output: list of list of float, one per input string, L2-norm == 1.0.
    Empty input yields ``[]``. Raises TypeError for non-list inputs or
    items that are not strings.
    """
    if not isinstance(texts, list):
        raise TypeError(
            "texts must be a list of str, got %s" % type(texts).__name__)
    for i, item in enumerate(texts):
        if not isinstance(item, str):
            raise TypeError(
                "texts[%d] must be str, got %s" % (i, type(item).__name__))
    if batch_size is None:
        batch_size = BATCH_SIZE
    else:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError(
                "batch_size must be int, got %s" % type(batch_size).__name__)
        if batch_size <= 0:
            raise TypeError("batch_size must be > 0, got %d" % batch_size)
    if not texts:
        return []
    items = list(texts)
    try:
        import torch
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    mdl = model if model is not None else load_model()
    try:
        mdl.eval()
    except Exception as exc:
        hub_cache = _hub_cache_dir()
        raise RuntimeError(
            "embedding model failed to enter eval mode: %s; HF cache: %s"
            % (exc, hub_cache)) from exc
    try:
        arr = mdl.encode(items, batch_size=batch_size,
                         convert_to_numpy=True,
                         normalize_embeddings=True,
                         show_progress_bar=False)
    except Exception as exc:
        cache = _hub_cache_dir()
        raise RuntimeError(
            "embedding encode failed for model %r: %s; HF cache: %s"
            % (MODEL_ID, exc, cache)) from exc
    return [[float(x) for x in row] for row in np.asarray(arr).tolist()]


def build_vectors(chunk_records, model=None, revision=None):
    """Build ``{chunk_id: [float, ...]}`` from chunk records.

    Uses each record's ``text`` field verbatim (storage is verbatim per
    the chunking contract). Returns a dict in input order. An empty
    record list yields ``{}``. Raises TypeError for malformed records.
    When this function loads the model itself, ``revision`` is passed
    to the loader (see ``load_model``); it is ignored when a model
    instance is supplied (the caller pinned it already).
    """
    if not isinstance(chunk_records, list):
        raise TypeError(
            "chunk_records must be list, got %s"
            % type(chunk_records).__name__)
    texts = []
    chunk_ids = []
    for i, record in enumerate(chunk_records):
        if not isinstance(record, dict):
            raise TypeError(
                "chunk_records[%d] must be dict, got %s"
                % (i, type(record).__name__))
        cid = record.get("chunk_id")
        text = record.get("text")
        if not isinstance(cid, str):
            raise TypeError(
                "chunk_records[%d]['chunk_id'] must be str, got %s"
                % (i, type(cid).__name__))
        if not isinstance(text, str):
            raise TypeError(
                "chunk_records[%d]['text'] must be str, got %s"
                % (i, type(text).__name__))
        chunk_ids.append(cid)
        texts.append(text)
    if not chunk_ids:
        return {}
    mdl = model if model is not None else load_model(revision=revision)
    embeddings = encode_texts(texts, model=mdl)
    return {cid: vec for cid, vec in zip(chunk_ids, embeddings)}


def retrieve(query, top_k, vectors, chunks_by_id, model=None):
    """Brute-force cosine retriever over pre-computed vectors.

    Per docs/RETRIEVAL_DESIGN.md section 9:
    - Cosine similarity = dot product of L2-normalized vectors.
    - Tie-break: ``(-score, chunk_id)`` (lexicographic ascending on ID).
    - Top-k truncation: exactly ``min(top_k, len(vectors))`` hits.
    - Contract dict: ``{"chunk_id", "score", "rank", "source"}``.
    - Source: ``"semantic"``. Rank: 1-based sequential int.

    Blank or whitespace-only query returns ``[]`` immediately.
    Empty vector store returns ``[]``.
    """
    if not isinstance(query, str):
        raise TypeError("query must be str, got %s" % type(query).__name__)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(
            "top_k must be int in [%d, %d], got %r"
            % (TOP_K_MIN, TOP_K_MAX, top_k))
    if not (TOP_K_MIN <= top_k <= TOP_K_MAX):
        raise TypeError(
            "top_k must be in [%d, %d], got %r"
            % (TOP_K_MIN, TOP_K_MAX, top_k))
    if not isinstance(vectors, dict):
        raise TypeError(
            "vectors must be dict, got %s" % type(vectors).__name__)
    if not isinstance(chunks_by_id, dict):
        raise TypeError(
            "chunks_by_id must be dict, got %s"
            % type(chunks_by_id).__name__)
    for cid, record in chunks_by_id.items():
        if not isinstance(cid, str):
            raise TypeError(
                "chunks_by_id key must be str, got %s"
                % type(cid).__name__)
        if not isinstance(record, dict):
            raise TypeError(
                "chunks_by_id[%r] must be a chunk dict, got %s"
                % (cid, type(record).__name__))
    if query.strip() == "":
        return []
    if not vectors:
        return []
    mdl = model if model is not None else load_model()
    qvec = encode_texts([query], model=mdl)[0]
    dim = len(qvec)
    try:
        qarr = np.asarray(qvec, dtype=np.float64)
    except Exception as exc:
        raise TypeError("query vector is not numeric: %s" % (exc,))
    ids = []
    rows = []
    for cid, vec in vectors.items():
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
    mat = np.stack(rows)
    scores = mat @ qarr
    order = sorted(range(len(ids)),
                   key=lambda i: (-float(scores[i]), ids[i]))
    hits = []
    for rank, i in enumerate(order[:top_k], start=1):
        hits.append({"chunk_id": ids[i], "score": float(scores[i]),
                     "rank": rank, "source": SOURCE_NAME})
    return hits


def get_dependency_versions():
    """Exact installed versions of the embedding stack (for metadata)."""
    import importlib.metadata as _md
    versions = {}
    for dist in ("sentence-transformers", "torch", "transformers",
                 "numpy", "scipy", "tokenizers", "huggingface_hub"):
        try:
            versions[dist] = _md.version(dist)
        except Exception:
            versions[dist] = "unknown"
    return versions
