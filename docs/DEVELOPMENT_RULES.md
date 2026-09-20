# AI Coding Rules — MizanIQ
AI-Powered Finance Document Intelligence System

> Rules that future coding agents MUST follow.
> Related docs: [PROJECT_SPEC.md](PROJECT_SPEC.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [ROADMAP.md](ROADMAP.md) · [DATASET_DESIGN.md](DATASET_DESIGN.md)

## Rule 1 — Read project documentation first

Before significant changes, read:

- docs/PROJECT_SPEC.md
- docs/ARCHITECTURE.md
- docs/ROADMAP.md
- docs/DEVELOPMENT_RULES.md
- docs/DATASET_DESIGN.md

## Rule 2 — Do not silently change architecture

Do not replace major technology or architecture decisions without approval.

Examples:

- DuckDB
- Qdrant
- hybrid retrieval
- visual retrieval architecture
- forecasting architecture

If there is a strong reason to change one:

STOP and report:

- current design
- problem discovered
- proposed alternative
- advantages
- disadvantages
- migration impact

Do not silently implement the change.

## Rule 3 — Work incrementally

Do not build multiple major phases in one task.

Each task should have a narrow objective.

## Rule 4 — Inspect before editing

Before modifying an existing component:

- inspect relevant code
- understand dependencies
- identify existing tests
- identify potential regressions

## Rule 5 — Preserve Git safety

Never:

- force push
- delete repository history
- reset destructive history
- delete unrelated user work

Do not commit automatically unless explicitly asked.

## Rule 6 — Secrets

Never commit:

- API keys
- tokens
- passwords
- private credentials

Use `.env`.

Provide `.env.example` with names/placeholders only.

## Rule 7 — Prefer deterministic tools where appropriate

Do not use LLMs for tasks that deterministic systems perform better.

Examples:

- SQL for arithmetic.
- Parsers for native text.
- Statistical models for forecasting.

## Rule 8 — Validate AI outputs

AI-produced structured output must be validated.

Pydantic or equivalent schema validation should be used.

Never trust unvalidated model-generated financial numbers.

## Rule 9 — Evidence-grounded answers

Answer generation must use provided/retrieved evidence.

If evidence is insufficient, the system should say so.

Do not fabricate missing financial information.

## Rule 10 — Source traceability

Extracted/retrieved information should preserve enough metadata to trace back to the original source.

Where applicable preserve:

- document_id
- filename
- page
- element/location
- language
- document type

## Rule 11 — Evaluation before scaling

Do not move from the representative development dataset to all ~150 documents until quality has been measured.

## Rule 12 — Performance is a requirement

Do not optimize only for accuracy.

Track:

- latency
- memory usage when relevant
- GPU requirements
- throughput
- ingestion time

But never sacrifice correctness blindly for speed.

## Rule 13 — Avoid unnecessary complexity

Do not introduce:

- agent frameworks
- microservices
- queues
- cloud infrastructure
- abstraction layers
- design patterns

unless the current problem genuinely requires them.

Start simple and split components as complexity becomes real.

## Rule 14 — Tests are part of implementation

A feature is not complete simply because the code runs once.

Add appropriate automated tests and/or evaluation tests.

## Rule 15 — Report honestly

After every implementation task report:

- files created
- files changed
- commands executed
- tests executed
- test results
- warnings
- unresolved issues
- assumptions

Never report success if tests failed.
