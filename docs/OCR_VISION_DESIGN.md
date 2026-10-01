# OCR / Vision Design — MizanIQ Phase 2 (Benchmark & Selection)

> How we choose the visual-document approach by measurement, not by hype.
> No production OCR/VLM is integrated here — this phase builds the
> escalation design, the benchmark harness, and the selection criteria.
> Related docs: [ARCHITECTURE.md](ARCHITECTURE.md) · [INGESTION_DESIGN.md](INGESTION_DESIGN.md) ·
> [PHASE_1_FREEZE.md](PHASE_1_FREEZE.md) · [EVALUATION_PLAN.md](EVALUATION_PLAN.md) ·
> [DEV_DOCUMENT_SPEC.md](DEV_DOCUMENT_SPEC.md)

## 1. Escalation architecture (intended, not yet built)

Native extraction remains preferred whenever reliable. The visual stage
is an escalation path, never the default:

```text
ingest_document(path)  [Phase 1, frozen]
    │
    ├─ NATIVE_OK ──────────────→ use native result (vision NEVER called)
    │
    ├─ REQUIRES_OCR ───────────→ OCR stage ──→ normalized Elements
    │   (DEV-010 style)          page-by-page    (Phase-1 model +
    │                                            meta.origin = "ocr:<model>")
    ├─ PARTIAL_NATIVE ─────────→ OCR stage for insufficient pages ONLY
    │   (mixed docs)             sufficient pages keep native Elements
    │
    ├─ REQUIRES_VISUAL ────────→ visual stage ──→ Elements + description
    │   (DEV-004/008/009)        (transcription AND/OR interpretation,
    │                             depending on fixture class below)
    │
    └─ FAILED / UNSUPPORTED ───→ no visual processing (garbage in, no GPU out)
```

Escalation granularity is the **page** (PDF) or **whole file** (raster):
`metadata.pages[].native_sufficient` (Phase 1) already tells the visual
stage exactly which pages need reprocessing. Re-running vision over
natively-sufficient pages is a measurable waste and is forbidden by
default; any exception must be justified by a benchmark delta.

## 2. Visual-processing responsibilities

1. **OCR text recovery** — verbatim text from raster content, with script
   fidelity (Arabic shaping must survive; reshaped glyphs are the input,
   logical strings the output).
2. **Layout-aware recovery** — reading order, blocks, and spatial
   association (which amount belongs to which row/label/card).
3. **Table recovery** — bordered and borderless tables → row/column
   structure with native numeric values, not flat text dumps.
4. **Chart/image interpretation** — trends, extrema, KPI cards, verdicts
   (a different task from transcription; see §7).
5. **Source-location preservation** — every recovered Element carries
   file + page (+ row/column/panel where applicable) so Phase-4+ citations
   and the evidence package stay grounded.

## 3. Candidate classes (roles mapped, sizes NOT locked)

| Class | Example family | Text extraction | Table/layout | Chart understanding | Visual retrieval |
|---|---|---|---|---|---|
| **A. Traditional OCR baseline** | Tesseract-class | ✓ (baseline) | structure-poor | ✗ | ✗ |
| **B. Modern OCR/doc-understanding** | PaddleOCR / PaddleOCR-VL | ✓✓ (AR+EN) | ✓ (layout-aware) | partial | ✗ |
| **C. General vision-language** | Qwen-VL family | ✓ (prompted) | ✓ (prompted) | ✓✓ | ✗ |
| **D. Visual-document retrieval** | ColQwen / ColPali-style | ✗ (not an OCR engine) | ✗ | ✗ (retrieves, doesn't read) | ✓ (Phase 6) |

Rules:

- **Do not force one model to solve every problem.** Class D scores
  page/chart recall; it is never graded on transcription. Class A is
  never graded on trend inference.
- **No exact model/size is hard-locked in this document.** The benchmark
  (§4–§7) exists precisely so sizes are chosen from measured deltas.
- Local status: no OCR engine is installed in this environment
  (Tesseract binary absent; see §8). The class-A baseline was PENDING
  until the first Kaggle experiment — that experiment has now run (see
  §10). Local installs are still out of scope; Kaggle remains the lab.

## 4. Benchmark fixtures (frozen `dataset_v0.1`, never modified)

Primary (visual processing is the only path):

- **DEV-004** — scanned Arabic receipt PNG · task: OCR (AR text, IDs,
  amounts under degradation) · 1038×1288.
- **DEV-008** — revenue line-chart PNG · tasks: OCR (title/labels) +
  visual understanding (trend, peak year, values) · 1600×1000.
- **DEV-009** — KPI dashboard JPG · tasks: OCR (cards/labels, mixed AR/EN)
  + visual understanding (KPI values, branch bars, budget verdict) ·
  1600×1000, q92.
- **DEV-010** — scanned balance-sheet PDF (1 raster page) · tasks: OCR +
  table recovery (Assets/Liabilities/Equity association) · rendered to
  PNG at fixed scale 2 (1191×1684 RGB, byte-deterministic via pypdfium2).

Secondary (native text exists → reference-grade comparison for
vision-vs-native agreement; rendered to PNG identically):

- **DEV-002** (Arabic native PDF), **DEV-003** (mixed invoice PDF).

`scripts/prepare_ocr_benchmark.py` exports all inputs into git-ignored
`data/benchmark/dataset_v0.1/` with a manifest (source file + sha256,
copy vs. rendered, render settings, language, role, tasks). Originals
are never modified; copies are byte-identical (sha-verified).

## 5. Derived benchmark truth (evaluation-side only)

`src/ocrbench/truth.py` derives a small scoring representation
programmatically from frozen GT — no manual duplication, no giant files:

```text
{ document_id, language, role, tasks,
  identifiers, text_anchors, numeric_values, fields,
  tables, chart, visual_elements, question_targets, source_location,
  unrendered_context }
```

- Text fixtures: identifiers + anchors + financial numerics + language +
  source location.
- Chart/visual fixtures: chart title/series/trend (DEV-008), KPI values +
  budget verdict inputs (DEV-009), labels, visual question targets.
- `unrendered_context` marks GT values that exist for comparative
  questions but are NOT rendered in the fixture and must never penalize
  OCR (currently: DEV-010 `equity_2019_context` — see
  [DEV_DOCUMENT_SPEC.md](DEV_DOCUMENT_SPEC.md) §DEV-010).
- **The production OCR service must never import benchmark truth.**
  Guarded by test (`ingestion` sources must not reference `ocrbench`).

## 6. Scoring (implemented in `src/ocrbench/`, stdlib-only)

Two independent layers — NEVER merged, averaged, or combined into one
overall score:

**LAYER A — RECOGNITION** ("did the model visibly recover it in raw
text?"):

- **identifier_text_accuracy** (`score_identifier_text`): strict
  substring presence of each expected identifier in `result["text"]`
  (whitespace/Arabic normalization only; hyphens, case, and digit values
  stay significant; no fuzzy matching; no GT-driven parser). A raw OCR
  engine with empty `fields` CAN score here.
- **numeric_text_exact_accuracy** (`score_numeric_text`): each expected
  value credited iff its canonical 2dp Decimal occurs among the amounts
  parseable from raw text. `24,371.25` == `24371.25` (grouping-insensitive);
  digit substitutions NEVER match; ±0.5% is NEVER applied at this layer.
  Means "the model saw this correct number somewhere."
- **table_label_text_recall** (`score_table_label_text`): each expected
  visible row label (e.g. `Cash`, `TOTAL ASSETS`, `Debt`) credited iff its
  normalized form occurs in raw text. No table object required.

**LAYER B — STRUCTURING** ("did the model attach it to the right
field/row?"):

- **identifier_structured_accuracy** (`score_identifiers`): exact
  field-value match in `result["fields"]` (legacy `identifiers` alias kept).
- **numeric_structured_exact_accuracy** (`score_numerics`, exact;
  legacy `numerics_exact` alias kept): 2dp `Decimal` equality per field
  after amount normalization; every miss listed expected-vs-got. Means
  "the model assigned this correct number to the correct field."
- **table_association_accuracy** (`score_table`): (row-label, value)
  associations — right digits on the wrong row = incorrect.

A model NEVER gets structured credit merely because the correct number
exists somewhere in its prose. A raw OCR engine is NEVER denied
recognition credit merely because it returned unstructured text.

- **Identifiers (structured):** normalized exact-match — strip outer whitespace only;
  hyphens and case are significant (`INV-2023-0106` ≠ `INV20230106`).
- **Numerics (structured):** 2dp `Decimal` equality after amount normalization
  (thousands separators, whitespace, SAR/`ر.س` tokens removed); every
  miss is listed expected-vs-got (EVALUATION_PLAN §3: numeric errors are
  their own error class). The ±0.5% OCR tolerance from EVALUATION_PLAN
  §3 is supported as an *explicit opt-in flag*, never the default, and is
  a secondary diagnostic only.
- **Anchors:** normalized-substring recall over `text_anchors`.
- **CER/WER:** Levenshtein-based, reported raw AND normalized.
  CER/WER computed against anchor concatenations are DIAGNOSTICS, not
  full-document OCR accuracy — no complete reference transcription
  exists for these fixtures, so they measure noise on the anchor slice
  only.
- **Arabic normalization** (`normalize.py`, documented rules):
  NFKC (folds Arabic-Indic ٠–٩ and Persian ۰–۹ to Western digits) →
  strip tatweel (U+0640) → strip diacritics (U+064B–U+0652 et al.) →
  unify alef forms (أإآٱ→ا) → collapse whitespace. Deliberately NOT
  folded: ة/ه, ى/ي, و/ؤ, punctuation — a model that confuses them is
  wrong, and folding would hide it. **Digit errors always survive
  normalization** (digits are compared exactly post-fold).
- **Tables (DEV-010):** never plain-text OCR alone. Scored as
  (row-label, value) associations: label recall → value accuracy →
  association accuracy (right digits on the wrong row = incorrect).
- **Charts/KPIs (DEV-008/009):** two separate scores — (A) recognition
  (visible text + numerics) and (B) semantic interpretation, graded
  deterministically from structured fields: peak year, trend keywords,
  KPI values, branch ranking, budget verdict + variance. Traditional OCR
  is never required to infer trends.

All candidates emit the standard result schema (§9 of the task spec —
`model, model_version, device, document_id, latency_ms, text, fields,
tables, visual_description, warnings`) validated by
`src/ocrbench/schema.py`, so one scorer compares every approach.

## 6b. Fair model comparison (recognition-first, then structuring)

Traditional OCR engines (class A) are judged FIRST on recognition
(Layer A): identifier/numeric text accuracy, anchor recall, label-text
recall. Empty `fields`/`tables` is expected — not a penalty at Layer A.

Layout / document-understanding models (class B) are judged on BOTH
layers: recognition AND structuring (correct field/row association).

Vision-language models (class C) may ADDITIONALLY be evaluated on
semantic interpretation (trend, verdict, ranking) — numeric faithfulness
still gates acceptance.

Consequences enforced by the scorer:

1. Correct digits in prose earn text credit but ZERO structured credit
   without the right field/row.
2. Recognition and structuring aggregates are reported side by side and
   NEVER averaged into one score.
3. The report columns `id_text / id_struct / num_text / num_struct /
   anchor / label_text / association` make the distinction obvious; `-`
   marks non-applicable metrics.

## 7. Experiment workflow (Kaggle GPU lab)

`notebooks/` holds a README and a minimal independent notebook: load
workspace PNGs → run ONE candidate → export schema-conformant JSON →
record latency/model/device. No weights, secrets, or production coupling
in the repo. First experiment (recommended): class-A Tesseract baseline
on DEV-004 + DEV-010 (cheap, CPU) to fix the floor; then class-B on the
same inputs; escalate to class-C only where B demonstrably fails
(DEV-008 trend, DEV-009 layout).

## 8. Model-selection criteria (how Phase 2 production gets chosen)

In order of importance: (1) Arabic text recovery, (2) numeric accuracy,
(3) identifier accuracy, (4) English recovery, (5) table-structure
recovery, (6) chart/KPI understanding, (7) latency, (8) GPU/VRAM needs,
(9) integration ease, (10) reproducibility, (11) licensing/deployment.
**Financial-number accuracy outranks prose aesthetics; the simplest
model that clears the gates wins** (EVALUATION_PLAN §§12–13: thresholds
TBD after baselines; one variable per ablation).

## 9. Explicitly NOT in this phase-step

Production OCR/VLM integration, embeddings, Qdrant, BM25, reranking,
answer generation, Streamlit, heavy weight downloads, CUDA setup.

## 10. First real Phase 2 OCR benchmark (measured — do not edit values)

Environment (Kaggle): PaddlePaddle 3.2.1, PaddleOCR 3.7.0, CUDA enabled,
Tesla T4. Pipeline: `PaddleOCRVL(pipeline_version="v1.6")`.

Real v1.6 API (locked in `src/ocrbench/paddle_adapter.py`):

- `predict(...)` returns an iterable of page result objects
  (`PaddleOCRVLResult`).
- Canonical text source is `page.json["res"]["parsing_res_list"]`
  (`block_label` / `block_content` / `block_bbox`), because it preserves
  block structure. `page.markdown` (also a dict) is fallback only.
- Observed labels: `paragraph_title`, `text`, `table`, `image`.
- Table `block_content` is HTML (e.g.
  `<table><tr><td>المبلغ</td><td>24,371.25 SAR</td></tr>…</table>`,
  parsed deterministically into list-of-row grids; `colspan=N` expands to
  N cells — first holds the text, the rest `""` — with no content
  duplication; `rowspan` is noted, not vertically expanded).
- Empty `image` blocks are ignored for recognition text (non-empty image
  content is kept as visual description only, never as text).
- `fields` stays `{}` for v1.6 outputs: Paddle emits recognition +
  structure, never semantic canonical fields. A row
  `المبلغ | 24,371.25 SAR` never becomes `fields["amount"]` on its own;
  recognition and table structure remain separate from downstream
  canonical semantic field extraction.

Measured via `scripts/score_ocr_results.py` (Layer A recognition and
Layer B structuring reported separately — never merged):

Tesseract (kaggle-CPU, `tesseract 4.1.1`, `eng` fallback —
`ara traineddata` missing):

- DEV-004: identifier text accuracy = 1.0, numeric text exact = 1.0,
  anchor recall = 0.4, structured fields = 0, latency ~603 ms
  (602.65 ms measured).
- DEV-010: numeric text exact = 1.0, anchor recall = 1.0, table label
  text recall = 1.0, table association = 0, latency ~856 ms
  (855.57 ms measured).

PaddleOCR-VL-1.6 (kaggle-GPU T4):

- DEV-004: identifier text accuracy = 1.0, numeric text exact = 1.0,
  anchor recall = 1.0, structured semantic fields = 0,
  latency = 8621.42 ms.
- DEV-010: numeric text exact = 1.0, anchor recall = 1.0, table label
  text recall = 1.0, table association accuracy = 1.0, structured
  semantic fields = 0, latency = 11274.39 ms.

Interpretation (do not collapse recognition, structure, and semantic
extraction into one score):

- Tesseract remains the fast raw-OCR baseline.
- Paddle materially improves Arabic coverage (DEV-004 anchor recall
  0.4 → 1.0) and structured table recovery (DEV-010 association 0 → 1.0).
- Paddle is roughly an order of magnitude slower on these fixtures
  (~0.6–0.9 s vs ~8.6–11.3 s).
- Semantic canonical field extraction is still a separate downstream
  concern (both engines report `fields = {}` on these runs).

Qwen was NOT run in this round (charts/KPI visual reasoning remain a
separate pending experiment).

## 11. Provisional Phase 2 routing policy (evidence-based candidate)

Current evidence-based candidate policy — documented here, NOT hardcoded
as production routing (the architecture still marks routing provisional):

1. Native parser first.
2. If native extraction is sufficient: stay native.
3. If scanned/simple text: lightweight OCR candidate may be sufficient.
4. If Arabic-heavy, layout-heavy, table-heavy, or complex scanned
   document: PaddleOCR-VL is the preferred current candidate.
5. Charts/KPI visual reasoning remain a separate pending experiment.

## 12. Visual-reasoning benchmark (locked design — DEV-008 + DEV-009)

Experiment question: does Qwen3-VL add meaningful visual-reasoning
capability beyond PaddleOCR-VL-1.6 for MizanIQ charts and KPI
dashboards? Run BOTH candidates on BOTH documents. No winner is
selected before measurement — this section locks the design only.

Locked decisions:

- Documents: DEV-008 (revenue trend chart 2015–2024, EN) and DEV-009
  (KPI dashboard Q4 2023, mixed AR/EN). Frozen fixtures and frozen truth
  are never modified to fit model output.
- Candidates: PaddleOCR-VL-1.6 (`PaddleOCRVL(pipeline_version="v1.6")`,
  official defaults — `chart_recognition` NOT overridden for the first
  visual pass; any future change is recorded in the identity, never
  hidden; never tuned per fixture) and Qwen3-VL.
- Qwen priority: `Qwen/Qwen3-VL-8B-Instruct` first;
  `Qwen/Qwen3-VL-4B-Instruct` ONLY if 8B cannot reasonably run on the
  Kaggle T4 (CUDA OOM, unsupported combination, memory ceiling). The 8B
  blocker is recorded in warnings/meta BEFORE any 4B use — never silent.
  Quantization needs a stop-and-report decision, never a silent change.
- Primary criterion: faithful numeric/chart/KPI understanding (exact
  visible numbers, exact labels, correct associations, faithful
  structure/trend, no hallucinations, no hidden recomputation, AR/EN
  handling). Secondary: latency. Prose quality is NOT the objective.
- Both models adapt to the SAME result schema
  (`model/model_version/device/document_id/latency_ms/text/fields/
  tables/visual_description/warnings`); raw outputs preserved separately
  (`paddle_raw_<DEV>.json`, `qwen_raw_<DEV>.json`); no hand correction.
- Notebook switches `RUN_PADDLE_VISUAL` / `RUN_QWEN_VISUAL` default
  False so Run All never triggers large downloads; sections Visual A
  (Paddle 008/009), Visual B (Qwen 008/009), Visual export.

DEV-008 scoring (frozen truth only):

- A. Visible fact recovery: chart title, year labels, visible values,
  units, legend/series names (`chart_labels`, `chart_display_text` over
  displayed "M" labels — canonical exact sums are NOT printed on the
  chart and are never demanded as text).
- B. Structured association: year -> displayed value, exact
  (`chart_association`); prose co-occurrence earns zero structured
  credit.
- C. Trend semantics: increasing/decreasing direction, highest/lowest
  displayed year, visible shifts (`chart_trend`); invented causes earn
  nothing and causal wording is flagged for review, never rewarded.
- D. Hallucination: invented years/values/series and unsupported claims
  counted independently (`chart_hallucinations`, `chart_claims`).

DEV-009 scoring (frozen truth + verified visible labels
`truth.DEV009_VERIFIED_LABELS`):

- A/B. KPI text/value recovery and label -> value association
  (`kpi_labels`, `kpi_display_text`, `kpi_association`) over displayed
  card/bar/panel strings.
- C. Budget panel canonicals: revenue 23,902,434.34, budget
  23,849,529.72, variance +52,904.62 (the panel prints the exact
  variance figure).
- D. Budget status: accuracy counts ONLY the visibly stated verdict
  ("Budget met" iff revenue actual >= budget; expense targets use actual
  <= budget). Recomputing an unquoted status earns no credit; refusing
  to infer beyond the visual is never penalized (`budget_status`).
- E. Mixed language: Arabic vs English required anchors scored
  separately (`anchors_arabic`, `anchors_english`).
- F. Hallucination: invented labels/values/conclusions counted
  independently (`kpi_hallucinations`).

Metric rule: every visual metric is independent
(`chart_label_recall`, `chart_numeric_exact_accuracy`,
`chart_association_accuracy`, `chart_trend_semantic_accuracy`,
`kpi_label_recall`, `kpi_numeric_exact_accuracy`,
`kpi_association_accuracy`, `budget_status_accuracy`,
`arabic/english_required_anchor_recall`, hallucination/claim counts,
`latency_ms`). No combined overall score, no averaging of correctness
with latency. Implementation: `src/ocrbench/metrics.py` (visual
section), `src/ocrbench/qwen_adapter.py` (8B/4B identity + fallback),
wired additively in `scripts/score_ocr_results.py`.
