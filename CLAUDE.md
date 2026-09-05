# demur — working notes

demur measures whether a tool-using LLM agent refuses correctly, stays inside policy, and does so at a known cost — and detects when a change makes any of those worse. A domain-blind library under `src/demur/`, plus one shipped specimen under `examples/governed_warehouse/` that exercises it end to end.

## Where the authority lives

- **`docs/spec.md`** — the specification. Read the relevant section before changing a model; §3 is the module map, §4 the data model and its diagram, §5 the extension points, §6 execution semantics.
- **`docs/build_plan.md`** — tickets are `D-xx` ids (`D-10 · Trajectory and manifest models`).
- **`docs/architecture.md`** — a class-by-class tour of what is built and why it is shaped that way.
- **`docs/provenance.md`** — what is borrowed, what is authored, and the client-data boundary.

If code and spec disagree, that is a bug in one of them. Fix both in the same change.

## Commands

```bash
uv sync
uv run pytest                  # 187 tests, ~0.2s — run them on every change
uv run ruff check .            # explicit ruleset in pyproject.toml, RUF100 on
uv run ruff format .
uv run pyright                 # src and tests; lists and string enums do not pass
```

CI runs `ruff check`, `ruff format --check`, `pyright`, and `pytest`. Everything goes through `uv run`.

## Architecture rules

1. **The boundary is enforced, not aspirational.** Nothing under `src/demur/` may import from `examples/`; `tests/test_boundary.py` walks the AST of every library module, relative-import escapes included. If the library appears to need domain knowledge, lift the concept into the library — do not import the specimen.
2. **The library knows no SQL and no tool names.** Which tool escalates is domain knowledge and arrives via the constraint set (see `Trajectory.ends_with_tool`), never hard-coded.
3. **Four protocols and one provider seam** (spec §5) are the sanctioned abstraction points. There is no `Constraint` protocol: the six rule types are a closed set. No registries, no config-driven dispatch, no compatibility shims.
4. **Readers use the record's API, not its fields.** The policy layer reaches a trajectory through `step_at`, `successful_calls_before`, `ToolCall.succeeded`, `ToolCall.result_fields` and `Completion.answered`, never by indexing `steps` or comparing `outcome.status`. The dataset loader reaches the policy through `abstention_ids` and `terminal_tools`. A new reader that needs more gets a method, not a field access.

## Data-model invariants

Each has a validator whose message explains itself and a test named after it. The argument for each lives in `docs/spec.md` §4 and §5; only the invariant is repeated here.

- **Records are deeply immutable.** `Record` is frozen with `extra="forbid"`, and `model_copy` re-validates. Sequences are tuples; JSON payloads are `FrozenJson`. Mapping defaults must be `default_factory=FrozenDict`, since pydantic does not validate defaults.
- **`reasoning_text` is a sibling of `response_text`**, on `Completion` and on `Message`. Nothing ties it to `usage.reasoning_tokens`. Thinking alone is not a turn.
- **Prompts are deltas.** `messages_appended` holds what arrived before the call, never its reply. A delta ending in an assistant turn is rejected.
- **`Completion` is the provider's reply; `LLMCall` adds `index` and the delta.** The provider never learns where in the run its reply sits.
- **`Sampling` is on both `RunManifest` and `LLMCall`.** The manifest holds what was configured, the call what was sent; a difference is evidence. A manifest rejects an unspecified temperature. `extra` may not shadow a named field.
- **`Usage` is all-or-nothing, and `None` is not zero.** Three disjoint prompt buckets plus output; reasoning is inside output. Disjointness is an adapter obligation, tested per adapter.
- **Identity is over source bytes**, via `record.sha256_hex`, never over `model_dump()`.
- **`final_answer` is present exactly when `terminal_state` is `completed`.**
- **`call_id` is required on request and dispatch**, and unique within a trajectory.
- **An `Interpretation` carries its own `reference_output`**, so a reading cannot exist without an answer. More than one interpretation means the instance must expect escalation.
- **Schema versions start at the first committed run.** Until then a breaking change rewrites the schema and the fixture. After, bump `SCHEMA_VERSION` and add a loader to `_LOADERS`. An optional field with a default never spends a version.
- **Constraints fail open on a missing argument.** Tools record every argument the policy reads, names qualified. `ConstraintSet.required_arguments` and `required_result_fields` publish both lists. A `Threshold` measurement that is present but not a number raises.
- **Constraints judge the subject step regardless of outcome, and count a prerequisite only if it succeeded.** Getting either half backwards makes T3 look like it improved the model.

## Tests

- `tests/conftest.py` provides the `trajectory` fixture — the canonical hand-written run: a blocked call, the recovery, then an answer.
- `tests/fixtures/trajectory-v1.json` is the golden file, byte-committed. If a test against it fails, the schema changed: bump the version and add a loader. Regenerate it only when the schema legitimately changed *and* the version moved with it.
- Name a test after the invariant it protects, and use the docstring to say why that invariant is load bearing. Tests here document reasoning as much as they check behaviour.

## Style

- Every docstring opens with a one-line summary of what the thing is or does, as in any library: `Return the step at index`, `Reject a naive started_at`, `One agent run against one instance`. The **why** follows after a blank line: the invariant, the failure it prevents, the alternative rejected. Nothing else: no restatement of the spec, history, ticket ids, project status, or what tests do.
- Write for a tired reader. Short sentences, one idea each, subject then verb. No asides in dashes or brackets. No rhetorical build-up. Plain words.
- Length limits, summary line included: module docstring 10 lines, class 6, method or validator 4, field comment 2. An argument that needs more goes in `docs/spec.md`, and the docstring cites the section.
- Error messages are two sentences: what is wrong, what to do.
- Example. Before: *"A run at 'whatever the provider defaults to' is not a measurement. Provider defaults change under stable model aliases, so a run recorded without a temperature cannot be reproduced even in principle, and a later run that differs cannot be attributed: the candidate changed, or the default did, and the artifacts do not say which."* After: *"Reject a manifest whose sampling has no temperature. `None` means the provider chose, and provider defaults move under stable aliases, so the run cannot be reproduced. `0` is fine."*
- Python 3.13, pydantic v2, 88-column ruff formatter.
- British spelling in prose, matching the spec.

## Status

Built: repository scaffold, CI and boundary test (D-01…D-04); the trajectory and manifest models with `schema_version`, `load_trajectory()`, and the golden fixture (D-10); the `Instance` schema and `sha256_hex` (part of D-20); the constraint engine with the six built-in types, `ConstraintSet` YAML loading, and the specimen's `constraints.yaml` (D-11); `Completion` as the provider's return type, ahead of D-14.

Still open on the critical path: D-13 tools, D-14 provider, D-15 agent loop, D-16 prompt rendering from the constraint set. The rest of D-20 — loading from disk, splits, dataset versioning — is unbuilt.
