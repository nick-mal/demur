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
uv run pytest                  # 97 tests, ~0.1s — run them on every change
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
- **Prompts are deltas.** `LLMCall.messages_appended` holds what arrived *before* the call; the call's own reply is `response_text` / `tool_calls_requested` and is rendered by `assistant_turn`. `prompt_at()` stops before that reply. Never store the reply in the delta — a validator rejects a delta ending in an assistant turn.
- **`Usage`: `None` is not zero.** Either the provider reported nothing (all four fields `None`) or the adapter normalised it, which means writing `cached_input_tokens=0` where there is no cache. Half-reported usage is rejected because it silently blanks `total_tokens`.
- **Instance identity is over source bytes**, never `model_dump()`. Hashing the parsed model would let a library change invalidate every committed baseline while the data stood still.
- **`final_answer` is present exactly when `terminal_state` is `completed`.** Escalation, budget exhaustion and failure all stop without answering.
- **`call_id` is required on both the request and the dispatch**, and unique within a trajectory. That pairing is how model intent is measured separately from what the guard allowed.
- **Committed artifacts are evidence.** Never edit a trajectory in `runs/` or `tests/fixtures/` to satisfy new code. Bump `SCHEMA_VERSION` and add a loader to `_LOADERS` keyed on the old version.

## Tests

- `tests/conftest.py` provides the `trajectory` fixture — the canonical hand-written run: a blocked call, the recovery, then an answer.
- `tests/fixtures/trajectory-v1.json` is the golden file, byte-committed. If a test against it fails, the schema changed: bump the version and add a loader. Regenerate it only when the schema legitimately changed *and* the version moved with it.
- Name a test after the invariant it protects, and use the docstring to say why that invariant is load bearing. Tests here document reasoning as much as they check behaviour.

## Style

- Comments and docstrings explain **why**, not what — the decision, the failure it prevents, the alternative rejected. Validator error messages teach the invariant to whoever trips them.
- Python 3.13, pydantic v2, 88-column ruff formatter.
- British spelling in prose, matching the spec.

## Status

Built: repository scaffold, CI and boundary test (D-01…D-04); the trajectory and manifest models with `schema_version`, `load_trajectory()`, and the golden fixture (D-10); the `Instance` schema and `content_hash` (part of D-20).

Still open on the critical path: **D-11** constraint engine (`policy/constraints.py`), then D-13 tools, D-14 provider, D-15 agent loop. The rest of D-20 — loading from disk, splits, dataset versioning — is unbuilt.
