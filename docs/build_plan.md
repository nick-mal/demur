# demur — build plan

Tickets in dependency order. Each states what to build, where it goes, and what is true when it is done. Read the matching specification section first; the ticket is the summary, not the source.

**Legend.** `[crit]` on the critical path — nothing downstream starts without it. `[par]` runnable in parallel with the current block.

---

## Block 0 — repository scaffold

**D-01 · Package skeleton** `[crit]`
Directory tree per specification §3. Every package gets `__init__.py`, including empty ones. `pyproject.toml` declaring the package with `mlflow`, `duckdb`, `pydantic`, `opentelemetry-sdk`, `fastapi`, `httpx`, `numpy`, `pytest`, `ruff`. Lockfile committed.
*Done:* `uv run python -c "import demur"` succeeds.

**D-02 · README**
What demur does (one paragraph), quick start, architecture tree. Findings listed as headings marked *not yet measured*. Status banner at the top.
*Done:* a reader who has never seen the project knows what it is within two minutes.

**D-03 · Provenance document**
Substrate source and licence terms; the authored-versus-borrowed split; how instances were produced.
*Done:* every artifact in the repository traces to a public source or to authorship recorded here.

**D-04 · CI and boundary test** `[crit]`
GitHub Actions running `ruff check`, `ruff format --check`, and `pytest` on push. `tests/test_boundary.py` walks the AST of every module under `src/demur/` and fails on any import from `examples/`, including relative-import escapes.
*Done:* green on main, and red when a deliberate bad import is added. **Verify the red case once — a guard that has never failed is not known to work.**

---

## Block 1 — library core and the example agent

**D-10 · Trajectory and manifest models** `[crit]`
`src/demur/trajectory.py` and `manifest.py`. Models per specification §4. Discriminated union on `kind` — without the discriminator, deserialisation can silently produce the wrong step type while a round-trip test still passes. `frozen=True` and `extra="forbid"` throughout. `schema_version` with a `load_trajectory()` dispatcher that raises a readable error on an unknown version.
*Done:* a hand-written trajectory round-trips to JSON and compares equal; a golden fixture is committed under `tests/fixtures/`; assigning to a loaded field raises; an undeclared key raises.
*Note:* this is the module everything else reads or writes. Build it in one uninterrupted block.

**D-11 · Constraint engine** `[crit]`
`src/demur/policy/constraints.py`. Six constraint types per specification §5, each returning `Violation | None` given a trajectory and a step index. `ConstraintSet` loads from YAML. Domain-blind: it knows tool names and orderings, never SQL.
*Done:* each type has a passing and a failing trajectory in tests, and the set loads from the example's `constraints.yaml`.

**D-12 · Warehouse fixtures**
`examples/governed_warehouse/warehouse/`. Load three BIRD train-split databases as three DuckDB schemas. Cast money to `DECIMAL` and dates to `DATE` at load — the source stores both as text and a silent bad cast returns a plausible wrong number. Subsample the largest database while preserving the full distribution of its status codes. Access matrix in YAML: 4–5 restricted tables, 8–10 denied columns with stated reasons, one scan-cost ceiling.
*Done:* `SELECT` across all three schemas returns rows; money and date columns carry proper types; five reference queries verified by hand return known-correct outputs; the built fixture is under 10 MB.

**D-13 · Four tools** `[crit]`
`examples/governed_warehouse/tools/`, against the `Tool` protocol. `run_query` takes a `dry_run` flag returning a projected scan cost computed from row counts and projected columns — **a static estimate, not a query planner.** `escalate` is idempotent per case id.
*Done:* each tool returns its documented shape; a duplicate `escalate` with the same case id produces one record.

**D-14 · Provider interface** `[crit]`
`src/demur/providers.py`. One protocol, one local implementation. The only abstraction permitted beyond the five in specification §5 — no registry, no config-driven dispatch, no compatibility shims.
*Done:* a completion round-trips and returns a populated `LLMCall` including token usage.

**D-15 · Agent loop** `[crit]`
Dispatch tools, record every step into a `Trajectory`, terminate on `escalate`, step-budget exhaustion, or the loop detector (three consecutive identical tool calls). No enforcement yet.
*Done:* one request produces a complete, serialised trajectory against fixtures, and a deliberately looping agent terminates as `abandoned`.

**D-16 · Enforcement T1** `[crit]`
`src/demur/policy/enforcement.py`. Render a `ConstraintSet` into system-prompt text. Prompt templates live in `examples/governed_warehouse/prompts/`, versioned; the rendered text is hashed into the manifest.
*Done:* the same constraint set produces stable prompt text, and its hash appears in the manifest.

---

## Authoring — parallel from the start of Block 1

Independent of code and bounded by elapsed thinking time rather than hours. Sequencing it after the implementation is what makes the first gate unreachable.

**D-20 · Instance schema** `[par]`
`src/demur/runner/dataset.py`. `Instance` per specification §4 including `interpretations`, `reference_outputs`, and `origin`. Content hashing; split loading.

**D-21 · Authoring guideline** `[par]`
`docs/authoring.md`, written *before* any instance. The rule: a case is ambiguous when the request fails to select among readings each defensible given the schema. Three worked examples, each verified against the actual data — a metric with two plausible column sources, a period with more than one defensible date anchor, a status vocabulary whose categories overlap.

**D-22 · Dataset v0 — 40 instances** `[par]`
All eight classes per specification §11. Author in batches of ten. Ambiguous cases cost roughly double: each needs its enumerated readings *and* one reference query per reading. The benchmark's own reference queries cover unambiguous cases only.

**D-23 · Splits and freeze** `[par]`
Development and test splits committed; test marked never-for-prompt-development; content hash asserted at evaluation time.

**D-24 · Versioning and corrections** `[par]`
Immutable versions, a defect issue log, corrections only as a new hash. Every run records the dataset version it ran against.

**D-25 · Dataset v1 — grow to 80** `[par]`

---

## Block 2 — first end-to-end slice

One thin slice through every layer. Deliberately incomplete: 40 instances, three scorers, one treatment, one model.

**D-30 · Scorer protocol and registry** `[crit]`
`src/demur/scoring/`. Registry keyed on `(id, version)`. A version change invalidates dependent baselines in code, not by convention.

**D-31 · `tool_sequence_valid`** — consumes constraint violations; domain-blind.

**D-32 · `result_correct`** — order-insensitive result-set equivalence. Explicit null-ordering and float-tolerance handling; these are what make naive comparison wrong.

**D-33 · `abstention_correct`** — deterministic against `len(interpretations) > 1`. Both error directions reported separately.

**D-34 · Evaluation runner** `[crit]`
`src/demur/runner/evaluate.py`. Execute instances through the agent, apply registered scorers, log per-item results and aggregates to MLflow. Parallel workers; per-instance trajectory writes so a crash leaves valid partial output. `--resume` skips instances already complete for every repeat.

**D-35 · Trajectory persistence** `[crit]`
Raw trajectories and scorer outputs to `runs/<run_id>/`, alongside the manifest. **Build this now, not at the end** — the replay guarantee depends on raw artifacts existing from the first run onward, and retrofitting it means the early runs are lost.

**D-36 · `demur eval` CLI** `[crit]`
One command: `--suite --model --repeats --resume`. Per-item results to stdout and MLflow.

**D-37 · First run committed** — T1, local model, dataset v0, results in `runs/`.

**D-38 · Integration slack**
Budgeted, not padding. The first end-to-end run reliably surfaces a mismatch between what the agent records and what the runner expects.

*Milestone: one command produces a scored run with per-item results over 40 instances.*

---

## Block 3 — depth

**Findings**

- **D-40** Treatments T2 (few-shot) and T3 (dispatch guard), strictly incremental over T1. **All three carry the identical T1 prompt** — stripping policy text from T3 would measure information removal rather than enforcement placement, and invalidates the comparison.
- **D-41** `access_policy_adherence` scorer.
- **D-42** `accepted_answer` — the four-way conjunction, and the `AcceptancePredicate` the example supplies to `economics.py`.
- **D-43** Enforcement metric set: invalid-attempt rate, recovery after a block, false rejection on legal paths, token and latency overhead.
- **D-44** Scorer versioning enforced against baselines.

**Regression machinery**

- **D-50** Baseline content-addressing over the manifest fields that determine comparability.
- **D-51** `tolerances.yaml` plus config hash recorded in every manifest. **Commit the bands before any candidate runs.**
- **D-52** Paired bootstrap with a one-sided non-inferiority criterion. Repeats collapse to a per-instance mean before resampling.
- **D-53** Per-item flips and the failure-category confusion table.
- **D-54** Three-class gating: capability, cost, reliability, with separate verdicts.
- **D-55 · Gate validation** `[crit]`
  Construct a deliberately degraded candidate and a noisy-but-equivalent one. The gate must flag the first *and stay silent on the second*. Most harnesses only ever test one direction; this bidirectional test is the one that makes the gate trustworthy.

**Service and reliability**

- **D-60** FastAPI `POST /ask` with typed schemas and a health endpoint. Thin — roughly 300 lines. Growth here is scope creep.
- **D-61** Docker Compose: API, warehouse, MLflow, one command.
- **D-62** OpenTelemetry spans emitted from the trajectory, convention version pinned, local collector.
- **D-63** Fault proxy — OpenAI-compatible shim.
- **D-64** Five injection profiles including the duplicate side-effecting call.
- **D-65** Fault run and results committed.
- **D-66** `economics.py` — observed cost per accepted outcome, decomposed, from a versioned price table.
- **D-67** Dashboard JSON committed.
- **D-68** Frontier model sweep, raw trajectories committed.

*Milestone: D-55 passes in both directions.*

---

## Block 4 — freeze and publish

Feature freeze first. Nothing from Block 3 crosses it.

- **D-70** Headline tables generated from committed raw results.
- **D-71** Both make targets verified — `reproduce-published-results` runs with the network disabled.
- **D-72** `docs/methodology.md`: treatment control with every prompt difference enumerated, the statistical procedure, tolerance rationale, and why the trajectory schema is owned rather than borrowed from the tracing layer.
- **D-73** `docs/validity.md` per specification §16.
- **D-74** README rewritten for a reader with eight minutes.
- **D-75** Clean-clone reproduction **on a machine that is not the development one**. This is the difference between a working repository and one that quietly stopped working.
- **D-76** Writeup published.

---

## Descope order

Decided in advance so it is not decided under pressure. Cut from the top.

1. Confidence-calibration appendix
2. D-67 dashboard JSON — traces already live in MLflow
3. D-25 dataset growth to 80 — publish on 40 and state the sample size
4. D-40 T2 only — the comparison survives as T1 against T3, weakened but interpretable
5. D-63–D-65 fault injection — costs the failure half of the economics finding
6. D-60–D-62 API, Compose, telemetry — reverts to an offline harness and forfeits the service boundary

Below item 6, nothing is cuttable without the project ceasing to be what the specification describes.