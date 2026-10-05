"""Deterministic stdlib BM25 keyword retriever (Phase 4 R2).

Implements docs/RETRIEVAL_DESIGN.md section 5 (BM25) and the ``bm25``
subset of the section 9 return contract over the frozen chunk inventory
built by ``retrieval.chunking`` (``data/retrieval/chunks.json``).

Scoring (frozen, never tuned)::

    score(D, Q) = SUM_t idf(t) * tf * (K1 + 1) / (tf + K1 * (1 - B + B * |D| / avgdl))
    idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))   (Lucene-style)

``K1`` / ``B`` are module constants pinned to the standard defaults.
Tokenization is owned solely by ``retrieval.chunking.normalize_text``
(design section 4); :func:`tokenize` here is a thin alias, never a
reimplementation. Query terms are deduplicated (each unique term
contributes once). ``|D|`` is the token count, ``avgdl`` the mean token
count over indexed chunks. Deterministic tie-break everywhere:
``(score desc, chunk_id asc)``.

Identifier exactness (design section 5: "identifiers must match
exactly"): a query part holding both alphabetic and numeric characters
(e.g. ``INV-2023-0106``) is identifier-like. Every detected identifier
must occur as a contiguous normalized-token run in a chunk for that
chunk to be eligible; eligible chunks then score with the plain BM25 sum
over all query terms. Queries without identifier-like parts behave as
pure keyword queries. Separator-insensitive by construction (``-`` vs
space vs ``/`` all tokenize away); token identity stays strict, so
near-matches (``0107`` vs ``0106``) and nonexistent IDs yield no hits.

Stdlib only: ``math`` plus the in-repo ``retrieval.chunking`` import.
No torch / numpy / sklearn / network access.
"""

import math

from retrieval.chunking import normalize_text

# Frozen BM25 constants (design section 5: THE standard defaults, never tuned).
K1 = 1.2
B = 0.75

# Mode name stamped on every returned hit (design section 9 return contract).
SOURCE = "bm25"

# top_k validation bounds (design section 9: 1 <= top_k <= 100, else TypeError).
TOP_K_MIN = 1
TOP_K_MAX = 100

__all__ = ["K1", "B", "SOURCE", "TOP_K_MIN", "TOP_K_MAX",
           "tokenize", "BM25Index"]


def tokenize(text):
    """Tokenize via the single tokenizer owner (chunking.normalize_text)."""
    return normalize_text(text)


def _strip_non_alnum(part):
    """Strip leading/trailing non-alphanumeric characters (builtins only)."""
    start = 0
    end = len(part)
    while start < end and not part[start].isalnum():
        start += 1
    while end > start and not part[end - 1].isalnum():
        end -= 1
    return part[start:end]


def _identifier_token_runs(query):
    """Detect identifier-like substrings in the raw query.

    An identifier-like substring is a whitespace-separated part holding at
    least one alphabetic and one numeric character (Unicode-aware, so
    Arabic-Indic digits count), e.g. ``INV-2023-0106``. Returns one
    normalized token list per detected identifier (via the single tokenizer
    owner, ``normalize_text``). An empty list means a plain keyword query.
    """
    runs = []
    for part in query.split():
        stripped = _strip_non_alnum(part)
        if not stripped:
            continue
        if (any(char.isalpha() for char in stripped)
                and any(char.isdigit() for char in stripped)):
            tokens = normalize_text(stripped)
            if tokens:
                runs.append(tokens)
    return runs


def _contains_run(doc_tokens, run):
    """Check a contiguous normalized-token run (exact identifier match)."""
    width = len(run)
    if width > len(doc_tokens):
        return False
    return any(doc_tokens[index:index + width] == run
               for index in range(len(doc_tokens) - width + 1))


class BM25Index:
    """In-memory BM25 index over chunk records.

    Chunk records are the dicts produced by ``retrieval.chunking``
    (only ``chunk_id`` and ``text`` are read here).
    """

    def __init__(self, chunks=None):
        self._doc_ids = []
        self._doc_tf = {}
        self._doc_len = {}
        self._doc_tokens = {}
        self._df = {}
        self._idf = {}
        self._N = 0
        self._avgdl = 0.0
        self._built = False
        if chunks is not None:
            self.build(chunks)

    def build(self, chunks):
        """Index a list of chunk records (replaces any previous state)."""
        if not isinstance(chunks, list):
            raise TypeError(
                "build expects a list of chunk records, "
                f"got {type(chunks).__name__}")
        doc_tf = {}
        doc_len = {}
        doc_tokens = {}
        doc_ids = []
        df = {}
        for position, record in enumerate(chunks):
            if not isinstance(record, dict):
                raise TypeError(
                    "build expects chunk-record dicts, "
                    f"got {type(record).__name__} at position {position}")
            chunk_id = record.get("chunk_id")
            text = record.get("text")
            if not isinstance(chunk_id, str):
                raise TypeError(
                    "chunk record at position "
                    f"{position} has non-str chunk_id: {chunk_id!r}")
            if not isinstance(text, str):
                raise TypeError(
                    f"chunk {chunk_id!r} has non-str text: {text!r}")
            if chunk_id in doc_tf:
                raise TypeError(f"duplicate chunk_id: {chunk_id!r}")
            tokens = normalize_text(text)
            term_freqs = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1
            doc_tf[chunk_id] = term_freqs
            doc_len[chunk_id] = len(tokens)
            doc_tokens[chunk_id] = tokens
            doc_ids.append(chunk_id)
            for token in term_freqs:
                df[token] = df.get(token, 0) + 1
        count = len(doc_ids)
        self._doc_ids = doc_ids
        self._doc_tf = doc_tf
        self._doc_len = doc_len
        self._doc_tokens = doc_tokens
        self._df = df
        self._N = count
        self._avgdl = (sum(doc_len.values()) / count) if count else 0.0
        self._idf = {
            term: math.log(1.0 + (count - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self._built = True
        return self

    def score(self, query):
        """Return ``{chunk_id: bm25_score}`` for chunks matching the query.

        Only chunks containing at least one query term appear (scores are
        strictly positive under the Lucene-style idf). When the query holds
        identifier-like parts, eligibility is additionally restricted to
        chunks containing every detected identifier as a contiguous
        normalized-token run; eligible chunks score with the unchanged BM25
        sum over all query terms. Empty/blank queries, unknown terms, and
        unmatched identifiers yield ``{}`` (never an error). A fresh,
        unbuilt index behaves as an empty index (``{}``).
        """
        if not isinstance(query, str):
            raise TypeError(
                f"score expects a str query, got {type(query).__name__}")
        terms = sorted(set(tokenize(query)))
        if not terms or not self._N:
            return {}
        id_runs = _identifier_token_runs(query)
        scores = {}
        for chunk_id in self._doc_ids:
            term_freqs = self._doc_tf[chunk_id]
            if id_runs:
                if not all(token in term_freqs for run in id_runs
                           for token in run):
                    continue
                if not all(_contains_run(self._doc_tokens[chunk_id], run)
                           for run in id_runs):
                    continue
            length = self._doc_len[chunk_id]
            norm = K1 * (1.0 - B + B * (length / self._avgdl
                                        if self._avgdl else 0.0))
            total = 0.0
            for term in terms:
                freq = term_freqs.get(term)
                if not freq:
                    continue
                idf = self._idf.get(term)
                if idf is None:  # unreachable when freq > 0; kept for safety
                    continue
                total += idf * (freq * (K1 + 1.0)) / (freq + norm)
            if total > 0.0:
                scores[chunk_id] = total
        return dict(sorted(scores.items()))

    def retrieve(self, query, top_k=10, chunks_by_id=None):
        """Rank chunks for ``query``: ``[{chunk_id, score, rank, source}]``.

        ``rank`` is 1-based; ``source`` is always ``"bm25"``; ordering is
        ``(score desc, chunk_id asc)``. ``chunks_by_id`` (optional
        ``{chunk_id: record}`` mapping) restricts ranking to its keys;
        ``None`` ranks the whole indexed corpus. Empty/blank queries and
        queries with no matching chunk yield ``[]``.
        """
        if not isinstance(query, str):
            raise TypeError(
                f"retrieve expects a str query, got {type(query).__name__}")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(
                f"top_k must be an int with 1 <= top_k <= 100, "
                f"got {top_k!r}")
        if not (TOP_K_MIN <= top_k <= TOP_K_MAX):
            raise TypeError(
                f"top_k must satisfy 1 <= top_k <= 100, got {top_k!r}")
        if chunks_by_id is not None and not isinstance(chunks_by_id, dict):
            raise TypeError(
                "chunks_by_id must be a dict or None, "
                f"got {type(chunks_by_id).__name__}")
        if not tokenize(query):
            return []
        scores = self.score(query)
        if not scores:
            return []
        if chunks_by_id is not None:
            scores = {cid: value for cid, value in scores.items()
                      if cid in chunks_by_id}
            if not scores:
                return []
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            {"chunk_id": chunk_id, "score": value, "rank": rank,
             "source": SOURCE}
            for rank, (chunk_id, value) in enumerate(ranked[:top_k], start=1)
        ]
