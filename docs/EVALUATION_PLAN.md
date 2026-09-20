# MizanIQ Evaluation Plan

AI-Powered Finance Document Intelligence System

> HOW we will know whether MizanIQ is actually good. Design only — no evaluation code yet.
> Related docs: [PROJECT_SPEC.md](PROJECT_SPEC.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [ROADMAP.md](ROADMAP.md) · [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) · [DATASET_DESIGN.md](DATASET_DESIGN.md) · [CANONICAL_DATA_MODEL.md](CANONICAL_DATA_MODEL.md)
>
> Status: DRAFT for architect review. All numeric thresholds are TBD until baselines are measured (§12).

## 1. Evaluation Philosophy

**We do not evaluate MizanIQ by saying the answer looks good.**

Every important subsystem has measurable tests with pre-registered gold targets. A demo answer may impress; only a scored benchmark convinces. Evaluation is separated by subsystem so a failure can be localized:

- ingestion
- OCR
- structured extraction
- text retrieval
- keyword retrieval
- reranking
- visual retrieval
- answer generation
- citations
- calculations
- forecasting
- latency / reliability

Each subsystem is scored against gold data (§2) with its own metrics (§§3–11), gated before scaling (§12), and compared fairly via ablations (§13).

## 2. Gold Data

### What "gold" means

Gold is the set of expected artifacts a system output is scored against. Gold must come from exactly three origins, in priority order:

1. **Canonical source records** — numeric answers recomputed from truth tables ([CANONICAL_DATA_MODEL.md](CANONICAL_DATA_MODEL.md)).
2. **Known generated-document content** — text/tables/charts fixed at generation time (render scripts + fixtures are the reference).
3. **Human verification where necessary** — a person confirms OCR transcripts, chart readings, cross-language equivalence, and abstention cases; reviewer + date recorded per item.

An LLM must NOT generate its own expected answers and then grade itself. LLM assistance may *draft* candidates (transcripts, QA pairs), but an item becomes gold only after origin-1/2 derivation or origin-3 sign-off. Unverified items are marked `unverified` and excluded from reported scores.

### Future gold artifacts (conceptual — do NOT create large files yet)

| Artifact | Content | Origin |
|----------|---------|--------|
| `gold_document_metadata` | doc_id, type, year, language, native/scanned, branch | generation fixture |
| `gold_financial_fields` | metric/value/currency/period + source location per field | canonical truth |
| `gold_questions` | question id, text (AR/EN), type, supporting doc/rows | truth + fixtures, human-verified |
| `gold_retrieval_targets` | per question: the chunk/page ids that count as relevant (graded: primary/acceptable) | fixtures + human check |
| `gold_citations` | per question: minimal sufficient source set (file + page) | fixtures + human check |
| `gold_visual_targets` | per visual question: correct page/chart/image id + expected reading | generation fixture + human check |

Relevance grading matters: `gold_retrieval_targets` distinguishes the *primary* evidence (the one page that answers the question) from *acceptable* supporting context, so Recall@K rewards finding the right page, not just any page mentioning the keyword.

## 3. Extraction Metrics

| Task | Recommended metrics | Notes |
|------|---------------------|-------|
| Native text extraction | normalized exact match, CER/WER on sampled pages | normalization = shared digit/punctuation/whitespace map (§5 of data model); report raw + normalized |
| OCR (scanned) | CER / WER, numeric-field accuracy | scored separately per language and per degradation level; Arabic-Indic digits normalized before comparison |
| Financial-field extraction | field-level precision / recall / F1 + numeric accuracy | a field counts correct only if metric + value + period + source ALL match |
| Table extraction | cell-level precision / recall / F1, table-structure accuracy (row/col alignment) | header mapping via approved vocabulary, not string equality |

Metric guidance:

- **Exact match** for identifiers (invoice IDs, branch codes) — no tolerance, ever.
- **Normalized exact match** for text/amounts after the shared normalizer (digits, currency tokens, whitespace).
- **Precision / recall / F1** at field and cell level — precision punishes invented fields (hallucinated rows), recall punishes missed ones; F1 summarizes but both components are always reported.
- **Numeric accuracy** with 2dp normalization for statement figures; the restricted ±0.5% relative tolerance applies ONLY to OCR-read values where scan noise is the tested variable.

For financial numeric fields, a wrong number is normally treated very seriously: numeric errors are reported as their own error class (not averaged away inside F1), and any evaluation report must list every numeric miss with expected vs. extracted values. A pipeline with high text F1 but wrong totals fails.

## 4. Retrieval Metrics

Definitions (per question, relevance from `gold_retrieval_targets`):

- **Recall@K** — fraction of relevant items retrieved in top K. Answers: "did we find the evidence at all?" Primary metric for the first-stage retriever (we need the evidence *in* the candidate set; reranking fixes order later).
- **Precision@K** — fraction of top-K that is relevant. Answers: "how much noise do we hand the reranker/LLM?" Matters for prompt focus and cost.
- **MRR (Mean Reciprocal Rank)** — 1/rank of the first relevant result, averaged. Answers: "how quickly do we surface one good answer?" Useful for single-evidence lookup questions.
- **NDCG@K (Normalized Discounted Cumulative Gain)** — rank-weighted relevance using graded (primary > acceptable) judgments. Answers: "is the *best* evidence at the *top*?" The ranking-quality metric of choice for explanation/multi-evidence questions.

Do NOT require all four everywhere: report **Recall@K + NDCG@K** as the standard pair (coverage + ranking), add Precision@K when prompt-noise analysis is needed, MRR for pure-lookup slices. K values (e.g., 5/10/20) are fixed per subsystem before experiments and identical across compared methods.

Comparison protocol: **semantic, BM25, and hybrid are evaluated separately before assuming hybrid is better.** Each runs against the same questions, same K, same gold targets; hybrid must beat both parents on the pre-registered primary metric (default: Recall@10) to justify its complexity. Per-language slices are always reported (a hybrid win driven solely by English while Arabic regresses is a failure — see §7).

## 5. Reranking Evaluation

The reranker is a *comparative* claim ("ordering improves"), so it is scored as a delta:

1. Freeze first-stage retrieval output (same candidates, same K) for every question.
2. Score retrieval metrics (§4) **before** reranking.
3. Apply reranker, score the same metrics **after** reranking on the identical candidate sets.
4. The reranker is accepted only if it measurably improves useful ranking quality (default primary: NDCG@5 on the reranked top-5, plus Recall@5 stability — reranking must not *lose* relevant items) **without unacceptable latency** (rerank latency budget TBD per §11; a +2pp NDCG gain at 10× latency fails).

Also report: latency distribution (p50/p95) of the rerank call, and per-category deltas (lookup vs. explanation vs. cross-language) — a reranker that helps English explanations but scrambles Arabic identifier ranking must be visible in the report, not hidden in an average.

## 6. Answer / Grounding Evaluation

Six separately scored dimensions — a single "answer quality" number is forbidden:

- **Answer correctness** — does the answer's factual content match gold (numbers from truth, explanations from event fixtures)? Numbers: recomputed check. Explanations: verified paraphrase/entailment against the event explanation, human-judged on a sample.
- **Groundedness** — is every factual claim in the answer supported by the retrieved evidence package? Unsupported claims are counted even if factually true (right answer, wrong basis = failure of the architecture's contract).
- **Citation correctness** — does each citation point to a real source (file + page) that exists in the corpus?
- **Citation completeness** — does the cited set cover all claims (no bare numbers) without citation stuffing (irrelevant sources cited for decoration)?
- **Insufficient-evidence behavior** — on abstention-designated questions, the ONLY correct behavior is an explicit decline ("the corpus does not contain…"); any fabricated answer is the worst error class. Also scored: false-abstention rate on answerable questions (over-cautiousness is a lesser but real defect).
- **Hallucination rate** — proportion of answers containing at least one claim contradicting gold or evidence. Reported overall and per language direction.

Foundational rule: **an answer can sound correct while citing the wrong source — that must count as an error.** Concretely: correctness-without-grounding scores 0 on groundedness and citation correctness, and the item is flagged in the hallucination/citation-error audit list. Leaderboards rank grounded-correct first; ungrounded-correct never outranks grounded-correct.

## 7. Multilingual Evaluation

Five separately reported categories — never one blended score:

| Category | Query | Evidence | What it isolates |
|----------|-------|----------|------------------|
| AR → AR | Arabic | Arabic docs | Arabic understanding end-to-end |
| EN → EN | English | English docs | baseline capability |
| AR → EN | Arabic | English docs | cross-lingual retrieval + AR generation |
| EN → AR | English | Arabic docs | cross-lingual retrieval + EN generation over AR evidence |
| Mixed | mixed AR/EN | mixed corpus | real operational queries |

Each category gets its own retrieval + answer + citation scores using informationally equivalent question pairs (same answer, same source rows — see data-model §5). Reporting rule: **no release or scaling decision may cite only the overall average; the minimum across the five categories gates progress.** A 90% overall score with 55% EN→AR is a multilingual failure wearing an overall-success costume.

## 8. Visual Evaluation

Visual retrieval is scored at two levels — finding and understanding:

1. **Correct page retrieval** — the source page containing the needed visual is in top-K (Recall@K over page ids from `gold_visual_targets`). Tests the visual index, independent of any VLM.
2. **Correct chart retrieval** — for chart questions, the specific chart/figure (not just its page) is retrieved.
3. **Correct image retrieval** — receipts, dashboard exports, invoice scans: exact-image Recall@K.
4. **Answer correctness from visual evidence** — end-to-end: given the visual question, is the produced value/reading correct AND cited to the visual source (file + page/figure)? A correct number cited to a text chunk that happens to mention it scores as a visual-path failure (right number, wrong evidence route).

VLM usage is itself an evaluated variable: report visual-QA accuracy with retriever-only (template/OCR reading) vs. with VLM inspection, plus VLM latency — the VLM is kept only where the delta justifies it, per the architecture's "VLM only when necessary" rule.

## 9. Calculation Evaluation

Exact deterministic calculations have **near-zero tolerance for logical errors**: the formula path (SQL over DuckDB records) is either right or wrong; rounding tolerance (±0.01 SAR after 2dp normalization) covers representation, never logic. A correct final number reached by wrong SQL on the dev set is still an error if the query logic is incorrect (review the generated SQL, not just its output on one data slice).

Evaluated operations (each with truth-recomputed expected values):

- single-value growth / difference (YoY deltas, absolute + %)
- sums and averages over periods (multi-year totals, monthly averages)
- ratios and margins (gross margin, net margin, expense ratios)
- budget variance (actual − budget, variance %; actual derived, never stored)
- branch comparisons (rank + gap, same metric/period across branches)

All checked directly against canonical data: the grader recomputes from truth tables independently of the pipeline's DuckDB contents, so a corrupted ingest (wrong numbers in DuckDB) cannot grade itself correct.

## 10. Forecasting Evaluation

Proper temporal validation only — no peeking, no single lucky split:

- **Holdout evaluation:** default final holdout = 2024 (12 months), touched once for final scoring.
- **Rolling-origin / walk-forward evaluation where useful:** multiple origins inside 2015–2023 for model selection and stability assessment; fixed origin calendar pre-registered before modeling.
- **Simple baseline first:** naive (last-value) → seasonal naive → ETS/SARIMA candidates. A complex model must beat the appropriate simple baseline to justify itself; seasonal-naive is the minimum bar for seasonal series.

Candidate metrics (report several; crown none universally):

- **MAE** — interpretable SAR-scale error; robust to outliers.
- **RMSE** — penalizes large misses (useful when stock-out-scale errors matter); sensitive to the 2020 shock window, so report with/without shock years.
- **sMAPE** — scale-free symmetric percentage; preferred over MAPE for cross-metric comparison.
- **WAPE** — scale-free, aggregation-safe; good headline metric across branches.

MAPE's pathology: when actual values approach zero (e.g., Dammam pre-opening months, sparse `other_income`), MAPE explodes or divides by zero — so MAPE is NOT a primary metric; sMAPE/WAPE cover the scale-free role. Metric choice per series is pre-registered (revenue/opex: WAPE + MAE; sparse series: MAE only), and every model comparison reports all pre-registered metrics — no metric-shopping after results.

## 11. Latency and Reliability

Measured separately per stage (instrumentation points, not guesses):

- ingestion time per page / per document (by format: native PDF, scanned PDF, DOCX, XLSX, CSV, image)
- retrieval latency (semantic / BM25 / hybrid separately)
- reranking latency (p50/p95 per call)
- SQL (DuckDB) latency per calculation query
- LLM generation latency (per answer, by output length band)
- visual-model latency (retrieval + VLM inspection separately)
- end-to-end question latency (by question route: text / visual / calculation / forecast)

Reliability tracked alongside:

- failure rate and retry rate (ingestion per document; QA per question)
- documents successfully processed (count + % per dataset version)
- resource requirements (RAM/VRAM/disk per stage; GPU necessity flags)

Do NOT set fake performance numbers before benchmarking. Reports present measured distributions (p50/p95, not just means) with hardware context. Latency never overrides correctness: a faster configuration that degrades grounded-correctness fails the gate regardless of speed.

## 12. Quality Gates

A QUALITY GATE is a pre-registered criterion that must be met before the project advances in scale or scope. Rationale: we do NOT move from 10 development documents to 150 just because the system "works" — the development set must meet agreed quality criteria first (DEVELOPMENT_RULES.md Rule 11).

Gate structure (each gate: metric + slice + threshold + measured value + pass/fail + date):

- **G1 — Extraction on DEV set:** field-level scores (§3) per format/language slice meet thresholds before any Phase 4+ retrieval tuning on new data.
- **G2 — Retrieval on DEV set:** hybrid-vs-parents comparison (§4) resolved; per-direction minima (§7) met before reranker adoption.
- **G3 — Grounded answers on DEV set:** grounded-correct rate, citation correctness, and abstention behavior (§6) meet thresholds before visual-retrieval expansion.
- **G4 — Calculation exactness:** §9 suite at (near-)zero logical error before forecasting begins.
- **G5 — Forecast validity:** baselines established, holdout protocol executed, complex-vs-simple justification recorded (§10) before Streamlit forecast UI.
- **G6 — Scaling gate:** v0.2 intermediate rehearsal meets adapted G1–G4 before `dataset_v1.0` generation.

Do NOT invent final threshold numbers yet — every threshold is **TBD until baseline experiments show reasonable ranges**. The procedure is fixed now (what is measured, on which slice, compared to what); the numbers are filled after first baselines run, proposed by the experimenter, and approved by the architect. A gate with a TBD threshold blocks advancement exactly like a failed gate.

## 13. Ablation / Comparison Testing

One important variable at a time. Planned comparison families (each: fixed questions, fixed gold, fixed K/holdouts; only the named variable changes):

- embedding model A vs. B (same chunking, same hybrid weights, same K)
- semantic vs. BM25 vs. hybrid (same embeddings, same K — §4 protocol)
- with vs. without reranker (same candidates — §5 protocol)
- OCR model A vs. B (same scanned pages, same CER/numeric-accuracy scoring)
- visual retrieval approach A vs. B (same visual questions, same page/chart recall scoring)
- forecast model vs. baseline (same origins, same holdout, same pre-registered metrics)

Each ablation produces a one-page record: variable changed, everything held fixed, primary metric delta, latency delta, per-slice notes (especially language directions), and the resulting keep/revert decision. No stacking of simultaneous changes when a comparison is inconclusive — rerun singly.

## Appendix A — Overengineering Review (Phase-0 Self-Check)

Reviewed both new documents against the rule: smallest model that robustly supports extraction, RAG, calculations, multilingual evaluation, and forecasting.

Deliberately excluded or simplified:

1. **No double-entry ledger / chart of accounts.** Balance-sheet figures come from an annual fixture, not debit/credit legs. Full accounting integrity is an ERP, not a document-intelligence benchmark.
2. **No depreciation schedule or fixed-asset register.** Folded into opex. Nothing in the question types needs asset-level depreciation.
3. **No inventory system.** COGS is a base input, not derived from stock movements. (Would add three tables for zero new evaluated capabilities.)
4. **No department monthly P&L table.** Department detail aggregates from transactions; a second P&L grain would create competing truths.
5. **No stored `actual_amount` in budgets.** Derived by aggregation — one truth, plus it exercises the required SQL path.
6. **No transaction line-item legs.** Line detail lives on invoice renderings; canonical truth stays at header + totals.
7. **No approval workflows, approvers, cost-center trees.** Realistic but unevaluated metadata.
8. **No multi-currency.** SAR-only V1 by architect approval; currency column omitted (dataset-level constant).
9. **No historical VAT rates in V1.** Flat 15% simplification, documented; real 5%/15% history deferred to a later advanced set.
10. **No dozens of business events.** ~8–12 curated events; each must earn its place by enabling a planned question type.
11. **No invented quality thresholds.** All TBD pending baselines — fake numbers would be worse than none.
12. **No universal forecasting metric.** Pre-registered per-series metric sets instead; prevents metric-shopping.

Residual risks to watch (not resolved by simplification): the annual balance-sheet fixture could drift from monthly P&L narratives (mitigate: fixture values cross-checked against truth aggregates at generation validation); sparse below-operating lines (`other_income` etc.) may be too thin to evaluate (mitigate: ensure at least 2–3 DEV/gold questions touch them or drop the columns before v1.0); branch-opening partial history (Dammam 2019) needs explicit NULL-vs-zero conventions in the generator spec.
