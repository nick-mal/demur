# demur — working notes

demur measures whether a tool-using LLM agent refuses correctly, stays inside policy, and does so at a known cost — and detects when a change makes any of those worse. A domain-blind library under `src/demur/`, plus one shipped specimen under `examples/governed_warehouse/` that exercises it end to end.

## Where the authority lives

- **`docs/spec.md`** — the specification. Read the relevant section before changing a model; §3 is the module map, §4 the data model and its diagram, §5 the extension points, §6 execution semantics.
- **Build plan** — tickets are `D-xx` ids (`D-10 · Trajectory and manifest models`). Currently at `.venv/build_plan.md`, which is **outside version control and will not survive recreating the venv** — worth moving under `docs/`.
- **`docs/provenance.md`** — what is borrowed, what is authored, and the client-data boundary.

If code and spec disagree, that is a bug in one of them. Fix both in the same change.

## Commands

```bash
uv sync
uv run pytest                  # 186 tests, ~0.2s — run them on every change
uv run ruff check .            # explicit ruleset in pyproject.toml, RUF100 on
uv run ruff format .
```

CI runs `ruff check`, `ruff format --check`, and `pytest`. Everything goes through `uv run`.

## Architecture rules

1. **The boundary is enforced, not aspirational.** Nothing under `src/demur/` may import from `examples/`; `tests/test_boundary.py` walks the AST of every library module, relative-import escapes included. If the library appears to need domain knowledge, lift the concept into the library — do not import the specimen.
2. **The library knows no SQL and no tool names.** Which tool escalates is domain knowledge and arrives via the constraint set (see `Trajectory.ends_with_tool`), never hard-coded.
3. **Five protocols and one provider seam** (spec §5) are the sanctioned abstraction points. No registries, no config-driven dispatch, no compatibility shims.

## Data-model invariants

These are easy to break and hard to notice, which is why each has a test and a validator with an error message that explains itself.

- **Records are deeply immutable.** Everything derives from `Record` (`record.py`): `frozen=True`, `extra="forbid"`, and `model_copy(update=...)` re-validates rather than skipping every validator. Sequence fields are tuples; JSON payloads use `FrozenJson` / `FrozenJsonObject`, which freeze recursively. Mapping defaults must be `default_factory=FrozenDict` — pydantic does not validate defaults, so a plain `dict` default is a live mutation hole.
- **`reasoning_text` is a sibling of `response_text`, not a shape `content` grows into.** Every provider reports thinking beside the content (`reasoning_content` on vLLM, `thinking` on Ollama, a `thinking` block on Anthropic), so `Message.content` stays `str`. It is on `Message` too, so `assistant_turn` carries it into a rebuilt prompt. Nothing ties it to `usage.reasoning_tokens` — Anthropic bills thinking while returning no text, vLLM returns text while reporting no count, and a validator demanding both would reject two correct adapters. Thinking alone is not a turn.
- **Prompts are deltas.** `LLMCall.messages_appended` holds what arrived *before* the call; the call's own reply is `response_text` / `tool_calls_requested` and is rendered by `assistant_turn`. `prompt_at()` stops before that reply. Never store the reply in the delta — a validator rejects a delta ending in an assistant turn.
- **`Sampling` (`sampling.py`) is on both `RunManifest` and `LLMCall`**, like `model` — the manifest holds what the run was *configured* with (and enters baseline comparability), the call holds what was actually *sent*. Divergence is evidence (a retry at a higher temperature, a fault profile), not an inconsistency. `None` means we did not specify it and the provider chose, so a manifest **rejects an unspecified temperature**; `0` is a fine answer. `extra` carries provider-specific knobs verbatim and may not shadow a named field. Manifest `seed` is the harness RNG; `sampling.seed` is the model's.
- **`Usage` fields are billing buckets, and `None` is not zero.** Three disjoint prompt buckets — `input_tokens`, `cached_input_tokens`, `cache_creation_input_tokens` — plus output; `reasoning_tokens` is inside output, not beside it. Either the provider reported nothing (all `None`) or the adapter normalised it, which means writing `0` into every bucket its provider has no *rate* for: a local model writes `0` to both cache buckets whatever its KV cache is doing. Half-reported usage is rejected because it silently drops tokens from `total_tokens`. Disjointness is an adapter obligation, unenforceable here — that is D-14's per-adapter tests.
- **Instance identity is over source bytes**, never `model_dump()`. Hashing the parsed model would let a library change invalidate every committed baseline while the data stood still.
- **`final_answer` is present exactly when `terminal_state` is `completed`.** Escalation, budget exhaustion and failure all stop without answering.
- **`call_id` is required on both the request and the dispatch**, and unique within a trajectory. That pairing is how model intent is measured separately from what the guard allowed.
- **Schema-version discipline starts at D-37, the first committed run.** Until then nothing published depends on `tests/fixtures/trajectory-v1.json`, so a breaking model change rewrites the schema and regenerates the fixture; `SCHEMA_VERSION` stays `1` because the schema has only ever had one published shape. **After D-37 a trajectory on disk is evidence**: never edit one to satisfy new code — bump `SCHEMA_VERSION` and add a converting loader to `_LOADERS` keyed on the old version. The test to apply is whether an older record still validates: an *optional* field with a default is additive and must **not** spend a version, a field joining a required set is a break and must.
- **Constraints fail open on a missing argument, so the tools carry the other half.** A key a call did not record supplies no value and matches no shape, so the rule is skipped rather than failed — a `run_query` recorded with only `sql` draws zero violations from the whole warehouse policy. Every tool must record every argument the policy reads (defaults filled in, derived values included) and must qualify table and column names, or a passing `check_access` on one table clears a same-named column on another. `ConstraintSet.required_arguments` and `required_result_fields` publish both namespaces to assert against — and a `Threshold` measurement that is present but not a number raises, because silently skipping it deletes the rule.
- **Constraints judge the subject step regardless of its outcome, and count a prerequisite only if it succeeded.** A blocked attempt is still an attempt — the invalid-attempt rate is model intent — and at dispatch time the subject has no outcome to consult. A blocked or errored earlier call did nothing, so it unlocks nothing. Getting either half backwards makes T3 look like it improved the model rather than merely stopped it.

## Tests

- `tests/conftest.py` provides the `trajectory` fixture — the canonical hand-written run: a blocked call, the recovery, then an answer.
- `tests/fixtures/trajectory-v1.json` is the golden file, byte-committed. If a test against it fails, the schema changed: bump the version and add a loader. Regenerate it only when the schema legitimately changed *and* the version moved with it.
- Name a test after the invariant it protects, and use the docstring to say why that invariant is load bearing. Tests here document reasoning as much as they check behaviour.

## Style

- Comments and docstrings explain **why**, not what — the decision, the failure it prevents, the alternative rejected. Validator error messages teach the invariant to whoever trips them.
- Python 3.13, pydantic v2, 88-column ruff formatter.
- British spelling in prose, matching the spec.

## Status

Built: repository scaffold, CI and boundary test (D-01…D-04); the trajectory and manifest models with `schema_version`, `load_trajectory()`, and the golden fixture (D-10); the `Instance` schema and `content_hash` (part of D-20); the constraint engine with the six built-in types, `ConstraintSet` YAML loading, and the specimen's `constraints.yaml` (D-11).

Still open on the critical path: D-13 tools, D-14 provider, D-15 agent loop, D-16 prompt rendering from the constraint set. The rest of D-20 — loading from disk, splits, dataset versioning — is unbuilt.
