# demur — architecture walkthrough

A guided tour of what is built, bottom up, and what each class is for. The specification (`spec.md`) is the authority on *what* the models are; this document explains *why the pieces are shaped the way they are* and how they fit. It covers the code as of the end of D-11 plus the review that followed it: four modules of data models and one module of logic. Everything else in specification §3 is unbuilt and is described only by its intended role at the end.

## 1. The shape in one picture

```
Instance ─────────── instance_id ──────────┐
(what was asked)                            │
                                            ▼
RunManifest ──────── run_id ─────────► Trajectory ◄──── ConstraintSet.evaluate()
(what the run was)                    (what happened)          │
                                                               ▼
                                                        Violation[]
```

Three records that never reference each other by object, only by identifier, so each can be read years after the others on its own. One engine, the constraint set, reads a trajectory and emits violations. Scorers, the dispatch guard and the prompt renderer are all consumers of that engine, and none of them exists yet.

The module dependency graph runs strictly downward and has no cycles:

```
record.py  ◄──  sampling.py  ◄──  trajectory.py  ◄──  policy/constraints.py  ◄──  runner/dataset.py
    ▲                                    ▲
    └────────────  manifest.py  ─────────┘ (sampling only; manifest and trajectory never import each other)
```

## 2. Layer 0 — `record.py`, the base everything stands on

| Class | Role |
|---|---|
| `Record` | Base class of every model in the library. Frozen, forbids unknown fields, and overrides `model_copy` so a copy is re-validated. "Records are evidence, not scratch space" is enforced here once rather than remembered per class. |
| `FrozenList`, `FrozenDict` | `list` and `dict` subclasses whose mutating methods raise. Pydantic's frozen flag stops attribute assignment but not `traj.steps[0].arguments["x"] = 1`; these close that hole. |
| `freeze()`, `FrozenJson`, `FrozenJsonObject`, `FrozenStrMap` | Apply those containers recursively to any JSON payload field, so nested tool arguments are immutable at every depth. |
| `Sha256`, `NonEmptyStr`, `TokenCount` | Validated scalar aliases. A malformed digest or an empty run id fails at construction rather than as a filesystem error hours into a run. |
| `sha256_hex()` | The one hash function. Instances, the constraint set and later the prompts are all addressed by their bytes on disk, never by the parsed model. |

## 3. Layer 1 — the three evidence records

### `trajectory.py` — what one agent run did

Read by every downstream component. The design decision that shapes the whole module is that **prompts are stored as deltas**: each call records what arrived before it, never its own reply, and the full request is rebuilt on demand.

| Class | Role |
|---|---|
| `Trajectory` | One run against one instance: a tuple of steps, a terminal state, the final answer, wall time. Validators enforce that step indices match positions, that call ids are unique, and that `final_answer` is present exactly when the state is `completed`. `prompt_at(i)` rebuilds the exact request for any LLM call; `ends_with_tool(name)` is the hook a `Terminal` rule uses to check that an escalated run really escalated. |
| `Step` | Discriminated union of `LLMCall` and `ToolCall` on `kind`. The discriminator is what stops JSON loading from guessing the wrong step type while a round-trip test still passes. |
| `Completion` | What a provider returned: `response_text`, `reasoning_text`, `tool_calls_requested`, finish reason, the `Sampling` and `Usage` that applied to this call. Nothing about where in the run it sits. `assistant_turn` renders the reply back into a `Message`. This is what `Provider.complete()` returns. |
| `LLMCall` | A `Completion` plus `index` and `messages_appended`, the delta that arrived before the call. The loop builds it; the provider never sees the index. |
| `Message` | One conversation entry, shaped per role: only assistant turns carry tool calls or reasoning, tool turns must name the call they answer, an empty turn is rejected as a recording bug. |
| `ToolRequest` | What the model asked for, before dispatch. Lives inside `Completion.tool_calls_requested`. Shares `call_id`, `name`, `arguments` and `raw_arguments` with `ToolCall` through a private base. |
| `ToolCall` | What the dispatcher did about that request, paired to it by `call_id`. The split between the two is how model intent is measured separately from what the guard permitted. `blocked_by` names the constraint that refused it. |
| `ToolOutcome`, `OutcomeStatus` | The result: `ok`, `error` or `blocked`. Blocked is deliberately not an error, because the tool never ran. |
| `Usage` | Token billing buckets: three disjoint prompt buckets plus output, with reasoning inside output. The all-or-nothing validator stops a half-reported count from silently shrinking a cost figure. `None` throughout means the provider said nothing. |
| `TerminalState` | Why the loop stopped. `escalated` is separate from `completed` because abstention is a correct outcome. |
| `load_trajectory()`, `SCHEMA_VERSION`, `_LOADERS` | The loading seam. Once runs are committed, an old file is read through a converting loader keyed on its version, never edited. |

### `sampling.py` — how a completion was requested

| Class | Role |
|---|---|
| `Sampling` | Temperature, top-p, top-k, max output, the model's seed, stop sequences, and an `extra` bag for provider-specific knobs recorded verbatim. It is on both the manifest and each call: the manifest holds what the run was *configured* with, the call holds what was actually *sent*. Divergence between them is a finding, not an inconsistency. |

### `manifest.py` — what the run was

Written once per run, and holding everything constant across it.

| Class | Role |
|---|---|
| `RunManifest` | Model, provider, treatment, sampling, dataset version and hash, system-prompt hash, constraint-set hash, scorer versions, library version, harness seed, repeats, start time. The two prompt hashes are the experimental control: identical `system_prompt_sha256` across T1, T2 and T3 proves no policy text was dropped from a treatment. Refuses an unspecified temperature and a naive timestamp. |
| `Treatment` | The three enforcement placements: prompt only, prompt plus few-shot, prompt plus dispatch guard. Strictly incremental. |

### `runner/dataset.py` — what the agent was asked

| Class | Role |
|---|---|
| `Instance` | One evaluation case: the request, fixture state, the constraint ids it is judged against, the expected resolution, interpretations and reference outputs, split and origin. `is_ambiguous` is simply "more than one interpretation", which is what makes the abstention finding definitional rather than judged. |
| `Expected`, `Resolution` | Whether the correct outcome is to answer or to escalate, with a free-text reason permitted only for escalation. |
| `Interpretation` | One defensible reading of the request, carrying its own `reference_output` as opaque JSON so the library never learns what SQL is. Ambiguity is constructed by enumerating these, not annotated by a judge. A reading cannot exist without its answer. |
| `Split`, `Origin` | `dev`/`test` and `authored`/`drafted`. Both bear on validity and are recorded per instance so a reader can check the mix. |
| `check_constraints_cover_ambiguity()` | Refuses an ambiguous instance that selects no abstention rule. The D-20 loader is intended to be its only caller. |

## 4. Layer 2 — `policy/constraints.py`, the one piece of logic

The premise: a policy is data with four consumers. The T1 prompt renders the rule descriptions, T2 picks exemplars that demonstrate them, the T3 guard evaluates them before each dispatch, and the scorer evaluates them over a finished run. One definition, so the treatments cannot drift apart, and its hash on the manifest proves they did not.

Evaluation is per step: every rule answers one question, *does the step at index `upto` violate this rule, given everything before it?* That framing is what lets the guard and the scorer share an implementation. The guard appends the call it is about to make and asks about that index; the scorer asks about every index.

Two conventions run through all six rule types and are easy to get backwards:

- **The subject step is judged whatever became of it.** A blocked attempt is still an attempt; the invalid-attempt rate is model intent, and at dispatch time there is no outcome yet.
- **A prerequisite counts only if it succeeded.** A blocked or errored earlier call did nothing, so it unlocks nothing.

| Class | Role |
|---|---|
| `_Rule` | Private base: an `id`, a human-written `description` the prompt will render, and the shared helpers that implement the two conventions above. `argument_dependencies` and `result_dependencies` publish which tool arguments and result keys a rule reads. |
| `RequiresBefore` | The workhorse ordering rule. `earlier_key`/`later_key` relate values across two calls; `earlier_when`/`later_when` let one tool gate itself; `satisfied_when` reads the earlier call's *result*, so a denied access check does not count as a passing one. |
| `Forbidden` | A tool is off limits, outright or in one argument shape. Not used by the shipped policy. |
| `Terminal` | After a successful call to this tool, every later step is a violation. This is how the library learns which tool escalates without hard-coding a name. |
| `Idempotent` | The tool may not succeed twice for the same value of `key`. |
| `Threshold` | Once a result field exceeds a ceiling, the only permitted next call is `else_action`; a successful `else_action` clears the breach. A measurement present but not numeric raises rather than being skipped. |
| `AbstainWhenUnderdetermined` | An LLM call that returns prose and no tool call is the agent answering, and under this rule that step is the violation. Reaches a run through `Instance.constraints`, since ambiguity is a property of the instance. |
| `ConstraintRule` | Discriminated union of the six on `type`, which is what lets YAML name them safely. |
| `Violation`, `ViolationKind` | The output: which rule, which step, a readable detail, and one *kind* per rule type so the failure-category confusion table keeps fixed columns while a policy renames rules. No `blocked` field, because the trajectory already knows. |
| `ConstraintSet` | The loaded policy: a version, the rules, and the hash of the file bytes the manifest records. `check_step` is the guard's call; `evaluate` is the scorer's; `select` picks the subset an instance names, refuses unknown ids, and carries no hash; `required_arguments` and `required_result_fields` publish the contract the tool layer must honour; `terminal_tools` is the seam the agent loop will use to know when to stop. |

**The tool contract.** The engine compares literal values in recorded arguments and can do nothing else without learning what a table is. A rule therefore fails *open* on a missing key: an argument a call did not record supplies no value and matches no shape, so the rule is skipped rather than failed. The other half of the contract is on the tools: record every argument the policy reads, defaults and derived values included, and qualify names that are only unique within a scope. `required_arguments` and `required_result_fields` exist so that contract is asserted mechanically.

## 5. How a run will move through this

Once D-13 (tools), D-14 (provider) and D-15 (agent loop) land:

1. The runner loads instances and the constraint set and writes a `RunManifest`.
2. For each instance, the agent loop calls the provider, which returns a `Completion`. The loop wraps it with the step index and the delta since the last call into an `LLMCall`.
3. For each `ToolRequest` in the reply, the loop builds a `ToolCall`. Under T3 it appends the call first and asks `ConstraintSet.check_step` about its index; if anything fires, the call is recorded as `blocked` with `blocked_by` naming the rule, and the tool never runs.
4. The loop stops on a terminal tool (`terminal_tools`), on step-budget exhaustion, or on three consecutive identical calls, and writes the `Trajectory`.
5. Scorers read the trajectory and its instance. `tool_sequence_valid` calls `evaluate` and separates blocked violations from executed ones via `traj.steps[v.step_index].blocked`.
6. The regression gate diffs scored runs across manifests whose comparability fields match.

## 6. Unbuilt modules, by role

| Module | Role |
|---|---|
| `policy/enforcement.py` | Renders a `ConstraintSet` into T1 prompt text and T2 exemplars; hosts the T3 dispatch guard. |
| `scoring/` | `Scorer` protocol, `ScoreResult`, and the versioned registry that invalidates baselines. |
| `runner/evaluate.py` | Executes instances, applies scorers, logs to MLflow. |
| `regression/` | Paired bootstrap, non-inferiority gate, flips and confusion table. |
| `faults/` | OpenAI-compatible proxy and the five injection profiles. |
| `telemetry/otel.py` | Emits `gen_ai.*` spans *from* a trajectory, not from inside the loop. |
| `economics.py` | Cost per accepted outcome from a versioned price table. |
| `providers.py` | The single model swap point. |
| `cli.py` | `demur eval` / `compare` / `replay`. |

The empty packages under `src/demur/` are placeholders for these and contain nothing.
