"""Isolated generic segmented-identifier eligibility experiment.

Compact mixed tokens remain ordinary BM25 terms. Normalization, scoring,
constants and public API are inherited from the accepted index.
"""
from retrieval.bm25 import (
    BM25Index, K1, B, tokenize, _strip_non_alnum, _contains_run,
)


def identifier_token_runs(query):
    runs = []
    for part in query.split():
        stripped = _strip_non_alnum(part)
        if (any(char.isalpha() for char in stripped)
                and any(char.isdigit() for char in stripped)):
            tokens = tokenize(stripped)
            if len(tokens) >= 2:
                runs.append(tokens)
    return runs


class ExperimentalBM25Index(BM25Index):
    """Only mandatory ID eligibility differs from accepted BM25Index."""

    def score(self, query):
        if not isinstance(query, str):
            raise TypeError(
                f"score expects a str query, got {type(query).__name__}")
        terms = sorted(set(tokenize(query)))
        if not terms or not self._N:
            return {}
        id_runs = identifier_token_runs(query)
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
                if idf is None:
                    continue
                total += idf * (freq * (K1 + 1.0)) / (freq + norm)
            if total > 0.0:
                scores[chunk_id] = total
        return dict(sorted(scores.items()))
