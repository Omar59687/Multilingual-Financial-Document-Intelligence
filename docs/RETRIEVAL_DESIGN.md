# Text Retrieval Design — MizanIQ Phase 4 (Text RAG)

> Phase 4 contract bible. Authoritative for chunking, normalization,
> retrieval APIs, index artifacts, and the question/gold procedure.
> Related: `docs/ROADMAP.md` (Phase 4) · `docs/ARCHITECTURE.md`
> (three memories) · `docs/EVALUATION_PLAN.md` (§§4/7/11/12/13) ·
> `docs/DEVELOPMENT_RULES.md` · `docs/CANONICAL_DATA_MODEL.md` (§5, §6)
> · `docs/DATASET_DESIGN.md` (§7 + isolation rules) ·
> `docs/DEV_DOCUMENT_SPEC.md` (per-doc fixtures/questions).

## 1. Phase 4 objective

Build the TEXT memory: deterministic chunking of the DEV corpus with
stable IDs and provenance-bearing metadata, a local deterministic
index, three first-stage retrievers (BM25 keyword, multilingual
semantic, RRF hybrid), a frozen dev/eval question set with graded gold
targets, and an objective G2 measurement. Reranking (Phase 5),
evidence packages (Phase 5), visual retrieval (Phase 6), answer
generation (Phase 7), and scaling (Phase 10) are explicitly OUT.

## 2. Corpus scope (frozen)

Indexed sources, in order:

1. Native text-bearing ingestion elements from all 10 DEV documents:
   `TEXT`, `HEADING`, `TABLE_CELL`, `SHEET_CELL`, `CSV_ROW`.
   Containers (`TABLE`, `TABLE_ROW`) and `IMAGE_REF` carry no text
   and are skipped (shapes, never content).
2. Paddle OCR texts (`data/benchmark/results/paddle_results.json`,
   `text` field) for scanned documents (DEV-004, DEV-010) — the
   approved Phase 2 OCR route output. Tesseract texts excluded
   (superseded route; documented).
3. Charts/standalone visuals (DEV-008, DEV-009 image content) carry
   no text: EXCLUDED from the text corpus (visual memory is Phase 6).

Ground-truth JSONs, canonical CSVs, the DEV manifest, and evaluation
question files live OUTSIDE the corpus and are never ingested,
chunked, or embedded (dataset-design isolation rules).

## 3. Chunking contract (frozen)

- `TEXT` / `HEADING`: one chunk each. TEXT chunks carry `section` =
  nearest preceding HEADING text in the same document, else None.
- `TABLE_CELL`: group by (page, table_index) [+ sheet where present]
  into ONE chunk; cells row-major as `" | "`-joined lines.
- `SHEET_CELL`: group by (sheet, row) into ONE chunk; header row 1
  supplies names, lines are `"header: value"` joined with `" | "`.
- `CSV_ROW`: one chunk per row; `"header: value"` pairs joined
  with `" | "` in mapping order.
- OCR texts: ONE chunk per scanned document (kind `ocr-text`;
  DEV-004 pageless/image-level, DEV-010 page 1).
- Chunk text is VERBATIM (never normalized in storage).
- Metadata per chunk: `document_id`, `filename`, `file_type`,
  `language` (document hint), `page`/`sheet`, `element_ids` (list),
  `kind` (`text`/`heading`/`table`/`sheet-row`/`csv-row`/`ocr-text`),
  `section`, `spans` (per-element `{element_id, start, end}` offsets
  into chunk text), char length.
- Chunk IDs: `CHK-<12 lowercase hex>` = first 12 of
  sha256 over canonical JSON (`sort_keys`, compact separators, UTF-8)
  of `{document_id, kind, index, text}`; `index` = per-document
  sequence in ingestion order. Deterministic across runs/platforms.
- Long TEXT elements are NOT split (traceability first); embedding
  truncation of long pages is a documented limitation (see §6).

## 4. Normalization contract (frozen, BM25 + token layer)

Storage stays verbatim; this normalization applies at tokenization
(query and index alike): NFKC → Latin casefold → Arabic-Indic digits
(`٠١٢٣٤٥٦٧٨٩`) to Western → Arabic punctuation `،؛؟` to `,,?` →
`\w+` Unicode word tokens. No stopword removal, no stemming
(multilingual asymmetry would bias language slices). Mirrors the
canonical §5 digit/punctuation rules; entity-splitting behavior is
identical for Arabic and Latin punctuation by construction.

## 5. BM25 contract (frozen, stdlib only)

Standard BM25 (`k1=1.2`, `b=0.75`, frozen — THE standard defaults,
never tuned): Lucene-style idf `ln(1 + (N - df + 0.5)/(df + 0.5))`
over §4 tokens; per-chunk avgdl normalization; deterministic
tie-break `(score desc, chunk_id asc)`. No new dependency
(`rank_bm25` NOT added). Exact identifiers (invoice/transaction IDs)
must match exactly — covered by token identity, asserted in tests.

## 6. Embeddings boundary (frozen)

- Provisional model: `paraphrase-multilingual-MiniLM-L12-v2`
  (sentence-transformers, CPU, eval mode — deterministic given
  frozen weights). NOT a hard lock (architecture forbids it):
  identity + HF revision hash recorded in index metadata; swap path
  kept open; BGE-M3 / Qwen-embeddings named challengers for a future
  embedding-benchmark task (EVAL §13 family). `all-MiniLM-L6-v2`
  rejected as default (English-only; would crater AR slices).
- Weights download from HF Hub at index-build time into the shared
  HF cache (outside the repo, no secrets, no API). No weight files
  committed. No fine-tuning ever (frozen weights — 2024-holdout and
  dev/eval-separation safe by construction).
- Vectors L2-normalized; cosine = dot product; brute-force scan
  (10 docs — Qdrant server VERIFIABLY DEFERRED to Phase 10 scale:
  unjustified service under project constraints); tie-break
  `(score desc, chunk_id asc)`.
- No new pip dependency recorded in `requirements.txt` (env-provided;
  exact versions recorded in index metadata).

## 7. Hybrid contract (frozen)

Reciprocal Rank Fusion, fixed `k=60` (documented constant, never
tuned): `score = Σ 1/(60 + rank)` over BM25 + semantic rankings
(full ranking depth, not top-K truncated). Missing from one side
contributes nothing. Deterministic tie-break `(score desc,
chunk_id asc)`. RRF chosen because it needs no score calibration or
fitted statistics (the minimal-need fusion).

## 8. Index artifacts (all under git-ignored `data/retrieval/`)

- `chunks.json`: chunk records in build order.
- `vectors.json`: `{chunk_id: [float, …]}` (model identity in meta).
- `index_meta.json`: model id + revision, chunker version/params,
  corpus manifest hash, counts, per-stage timings, generated_at.
- Determinism: two builds → identical chunks + vectors
  (`generated_at` excluded from the comparison).

## 9. Retrieval API (frozen, Phase-5-facing return contract)

`retrieve(query, top_k, mode)` with `mode ∈ {bm25, semantic, hybrid}`
returns ranked `[{chunk_id, score, rank, source}]` (`source` = mode
name) plus `get_chunk(chunk_id)`. Deterministic ordering guaranteed.
`top_k` validated (`1 ≤ top_k ≤ 100`, else `TypeError`). Empty/blank
queries → empty list (never an error, never fabricated hits).
Malformed inputs → `TypeError`; missing index artifacts →
explicit `FileNotFoundError`-family message (never silent).
Latency measured per call (EVAL §11 reporting; no thresholds).

## 10. Question set + gold procedure (frozen)

- ~24 questions from DEV fixtures (per-doc example questions +
  GT question targets): identifier lookups (INV/TX IDs), factoids
  (statement/balance/receipt figures), explanations (event passages
  in DEV-005). Languages: EN (EN docs), AR (AR docs — short,
  fixture-following), mixed-doc EN. NO cross-direction pairs (need
  bilingual equivalence judgment → human-gate backlog).
- Fixed IDs `Q-DEV-*` / `Q-EVAL-*`; the EVAL split is held out from
  day one (never used for tuning — there is no tuned surface in
  Phase 4: BM25/RRF constants frozen; still enforced structurally).
- Gold targets per question: chunk IDs with `primary` (answers it)
  vs `acceptable` (supporting context) grading, derived from
  fixtures + chunk inventory — human CHECK required (EVAL §2
  origin-3) at the terminal gate.
- G2 measurement: Recall@10 (primary) + Recall@5 + NDCG@10, same
  questions/same K/same gold for all three modes (§4 protocol);
  per-language slices + minimum reported (§7 rule — no blended-only
  claims); latency p50/p95 per mode (§11). Thresholds TBD →
  architect approval (G2 gate, same pattern as G1).

### 10.1 Approved numeric NDCG relevance gains (human architect decision)

Approved by **Omar Ahmad**, human/architect, on **2026-10-05**:

| Relevance | NDCG gain |
| --- | --- |
| primary | 2 |
| acceptable | 1 |
| irrelevant / non-gold | 0 |

This numeric mapping is architect-approved for Phase 4 retrieval evaluation
and is no longer provisional. It supplements the authoritative requirement
primary > acceptable with a simple deterministic mapping that gives stronger
credit to primary evidence. The accepted scorer already uses these values;
no numeric implementation change is required. Earlier scorer comments marking
the gains provisional/pending approval describe the pre-decision state and
are superseded by this explicit human decision.

The G1-approved frozen gold set contains only primary targets. Approval of
this gain mapping does not change any gold question, target, relevance grade,
or G1 sign-off, and does not retroactively change primary-only judgments.
G1 integrity is confirmed: DEV 12/12, EVAL 12/12, overall 24/24 valid sign-offs
by Omar Ahmad dated 2026-10-05; invalid/incomplete sign-offs NONE.

The post-G1 R6 measurement refresh is now authorized and READY, but has not
been performed as part of this policy record. G2 acceptance thresholds remain
TBD and must not be invented. G2 and Phase 5 remain NOT STARTED.

## 11. Non-goals enforced here

No reranker, no evidence packages, no citations generation, no answer
generation, no visual retrieval, no Qdrant service, no API/secret
dependencies, no embedding fine-tuning, no Phase 3 modifications, no
GT/canonical reads at retrieval runtime, no 2024-special-casing
(retrieval trains nothing; the holdout binds modeling phases).
