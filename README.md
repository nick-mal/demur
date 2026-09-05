# ! project is in active development

# 1. What demur does

demur measures whether a tool-using LLM agent **refuses correctly, stays inside policy, and does so at a known cost** — and detects when a change makes any of those worse.

- **Trajectory schema** — canonical, serialisable record of what the agent did: every LLM call, every tool call, arguments, outcomes, tokens, timing.
- **Constraint engine** — ordering and permission rules as data, evaluated against a trajectory into typed violations. One constraint set drives the system prompt, the exemplars, the runtime guard, and the scorer: one definition, four consumers.
- **Three enforcement strategies** over that set, so the incremental effect of runtime enforcement is measured rather than assumed.
- **Scorer protocol and registry** with mandatory versioning; a version change invalidates dependent baselines automatically.
- **Evaluation runner** on MLflow — per-item results, aggregates, OpenTelemetry traces.
- **Regression detection** — paired bootstrap, one-sided non-inferiority, tolerance bands fixed before candidates run, per-item flips, failure-category confusion table.
- **Fault proxy** — OpenAI-compatible shim injecting upstream failures so recovery behaviour and its cost are measurable.
- **Cost accounting** — cost per accepted outcome, acceptance predicate supplied by the caller.

# 2 Install and run
```bash
git clone https://github.com/nick-mal/demur && cd demur
uv sync
docker compose up -d        # API + DuckDB warehouse + MLflow

demur eval --suite examples/governed_warehouse --model local:qwen3 --repeats 3
demur compare --baseline runs/baseline-v1 --candidate runs/latest
make reproduce-published-results
```

# 3 Architecture (Presumed)
```
demur/
  src/demur/                    ← the library. No domain knowledge. No SQL.
    trajectory.py               Trajectory, Step, ToolCall, LLMCall, Usage
    policy/constraints.py       constraint types + evaluation → Violation[]
    policy/enforcement.py       T1/T2/T3 strategies over a ConstraintSet
    scoring/protocol.py         Scorer protocol, ScoreResult
    scoring/registry.py         versioned registration, baseline invalidation
    runner/dataset.py           Instance loading, content hashing, splits
    runner/evaluate.py          MLflow-backed execution + scoring
    regression/bootstrap.py     paired bootstrap over instances
    regression/gate.py          non-inferiority decision, tolerance config
    regression/report.py        flips, failure-category confusion table
    faults/proxy.py             OpenAI-compatible shim
    faults/profiles.py          injection profiles
    telemetry/otel.py           gen_ai.* spans, pinned conventions
    economics.py                cost per accepted outcome
    providers.py                the single LLM swap point

  examples/governed_warehouse/  ← the shipped specimen
    tools/                      describe_schema, check_access, run_query, escalate
    warehouse/                  public schema, seed data, access matrix (YAML)
    constraints.yaml            the six policy rules as data
    dataset/                    instances, interpretations, splits, hashes
    scorers/                    result_correct, access_policy_adherence, abstention_correct
    api/                        FastAPI POST /ask — thin, ~300 lines
    runs/                       committed raw trajectories + scorer outputs

  docs/                         motivation, methodology, provenance, validity
```

# Ultimately goal of this project is to measure the following (yet untested):

## dispatch-layer enforcement prevents unauthorised side effects while preserving valid completion, at a quantified overhead.

## abstention quality under ambiguity.

## unit economics under injected failure. Observed cost per accepted answer, with and without faults.