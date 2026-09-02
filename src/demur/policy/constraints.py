"""Policy as data: ordering and permission rules, evaluated into typed violations.

One constraint set drives four consumers — the system prompt (T1), the few-shot
exemplars (T2), the runtime dispatch guard (T3), and the `tool_sequence_valid`
scorer. That is the reason the rules are data rather than code in each of the
four: a policy expressed four times is a policy that drifts three ways, and the
enforcement-placement finding compares treatments that must be enforcing exactly
the same thing. `constraint_set_sha256` on the manifest is how a reader checks
that they were.

**Domain-blind.** A constraint knows tool names, argument names and orderings.
It does not know that `run_query` takes SQL, that `orders` is a table, or that a
projected scan cost is measured in rows. Argument matching is literal equality
against values named in the rule; nothing here parses a value or understands
what it means. That is the boundary in specification §3, and it is why the same
engine works for a warehouse specimen and for whatever comes after it.

**Not a configuration DSL** (specification §15). Every field on every rule is a
tool name, an argument name, a literal value, or a number. There is no
expression language, no predicate syntax, no dispatch on strings. Six types,
fixed, each a plain typed model — if a policy needs something they cannot say,
the answer is a seventh type argued for on its merits, not an escape hatch.

**Evaluation is per step.** `check(traj, upto)` asks one question: does the step
at index `upto` violate this rule, given everything before it? Framing it that
way is what lets the dispatch guard and the scorer share an implementation
rather than agree by convention — the guard appends the call it is about to make
and asks about that index; the scorer asks about every index in a finished run.

**Whose outcome counts.** The step under judgement is judged whatever became of
it: an attempt the guard blocked is still an attempt, and the invalid-attempt
rate is defined as model intent. Earlier steps count as satisfying a
prerequisite only if they *succeeded* — a `describe_schema` that errored
described nothing, and a call the guard refused never ran at all. The
asymmetry is deliberate and it is also what makes the guard and the scorer
agree: at dispatch time the subject's own outcome does not exist yet.

**A rule cannot see an argument that is not there, and it fails open in both
directions.** A key the recorded call does not carry supplies no value to match
(`earlier_key`, `later_key`) and matches no shape (`earlier_when`, `later_when`,
`when`), so an omitted argument means "requires nothing" on one side and "this
rule does not apply" on the other. Neither can be fixed here without the engine
inventing a requirement the tool itself would have accepted, which would put the
policy engine into the false-rejection metric.

It is a real hole all the same: a `run_query` recorded with only `sql` draws
zero violations from the whole eight-rule warehouse policy. Closing it is a
**contract on the tool layer** — every tool must record, on every call, each
argument the policy reads, filling in defaults it applied and values it derived
rather than leaving the key absent. `ConstraintSet.required_arguments` exists so
D-13 can assert that mechanically instead of the contract being remembered.

`ToolCall.arguments` is the right place for that normalised form and there is no
tension with recording verbatim intent, because intent is recorded elsewhere:
`ToolRequest.arguments` on the preceding `LLMCall` is what the model asked for,
the `ToolCall` is what was dispatched, and `call_id` pairs them. The policy
judges the dispatch.

**Literal equality is the only comparison, so distinctions the tool does not
encode do not exist.** A column named `salary` matches a column named `salary`
whichever table each came from. The engine cannot do better without learning
what a table is; the second half of the same tool contract is therefore that
values which are only unique within a scope are recorded qualified —
`employees.salary`, not `salary`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Self, runtime_checkable

import yaml
from pydantic import Field, model_validator

from demur.record import (
    FrozenDict,
    FrozenJsonObject,
    NonEmptyStr,
    Record,
    Sha256,
)
from demur.trajectory import LLMCall, OutcomeStatus, Step, ToolCall, Trajectory


class ViolationKind(StrEnum):
    """The failure-category label a violation carries.

    One kind per constraint type, and deliberately *not* one per constraint:
    the failure-category confusion table in specification §7 compares how
    failure modes migrate between a baseline and a candidate, and its columns
    have to stay fixed while a policy gains or renames rules. The rule that
    fired is `Violation.constraint_id`; this is the shape of the failure.
    """

    MISSING_PREREQUISITE = "missing_prerequisite"
    FORBIDDEN_CALL = "forbidden_call"
    AFTER_TERMINAL = "after_terminal"
    DUPLICATE_CALL = "duplicate_call"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    FAILED_TO_ABSTAIN = "failed_to_abstain"


class Violation(Record):
    """One rule broken at one step.

    No `blocked` field. Whether the guard stopped the offending call is
    `traj.steps[step_index].blocked`, and storing it here as well would let a
    replay and the enforcement metrics read different things from the same
    step — the same reason `LLMCall.assistant_turn` is derived rather than
    stored.

    `detail` is written for whoever reads the failure, not for a parser. It
    names the values that were missing, duplicated or over the ceiling, because
    a violation that only says which rule fired sends its reader back to the
    trajectory to work out why.
    """

    constraint_id: NonEmptyStr
    kind: ViolationKind
    step_index: int = Field(ge=0)
    detail: NonEmptyStr


@runtime_checkable
class Constraint(Protocol):
    """The extension point of specification §5.

    The six built-ins below implement it. A caller with a rule the built-ins
    cannot express implements this instead of reaching for a configuration
    escape hatch; what it cannot do is arrive from YAML, which is the point —
    a rule that is code is reviewed as code.
    """

    id: str

    def check(self, traj: Trajectory, upto: int) -> Violation | None: ...


def _literal(value: Any) -> str:
    """A JSON value as a hashable, type-distinguishing string.

    Everything goes through `json.dumps`, including strings, so that the
    integer `1` and the string `"1"` do not compare equal. A table named by one
    call and a row count returned by another must never satisfy each other's
    prerequisite just because they render the same.
    """

    return json.dumps(value, sort_keys=True)


def _key_values(source: Mapping[str, Any], key: str) -> frozenset[str]:
    """The value or values a call carries at `key`, as a set.

    Scalars and lists are both accepted and both become sets, because the same
    concept appears in both shapes across a tool suite — one call describes a
    single table, another queries several. Collapsing them here is what lets one
    rule relate the two without knowing what a table is.

    A key the call does not carry yields the empty set, which satisfies nothing
    and requires nothing. Making an argument mandatory is the tool schema's
    job; a constraint that invented the requirement would reject calls the tool
    itself would have accepted.
    """

    if key not in source:
        return frozenset()
    value = source[key]
    items = value if isinstance(value, list | tuple) else [value]
    return frozenset(_literal(item) for item in items)


def _matches(source: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Whether `source` carries every key in `expected` with an equal value.

    A partial match, not an equality test: a rule that named every argument a
    call may carry would break the first time a tool grew an optional one.

    Compared through `_literal`, so `True` and `1` are different values here
    exactly as they are everywhere else in this module. Python would call them
    equal, and `satisfied_when: {allowed: true}` would then be satisfied by a
    tool that returned `allowed: 1` — a rule about permission answered by a
    count.

    A key `source` lacks never matches. That is the fail-open direction
    described in the module docstring: the rule is skipped rather than
    satisfied, and the tool contract is what keeps the key present.
    """

    return all(
        key in source and _literal(source[key]) == _literal(value)
        for key, value in expected.items()
    )


def _number(value: float) -> str:
    """A measurement as a reader would write it.

    `%g` turns a scan cost of sixty million into `6e+07`, which is the wrong
    end of a violation message: whoever is reading it is comparing two
    magnitudes by eye, and exponent notation is exactly where that goes wrong.
    """

    if not math.isfinite(value):
        # A tool reporting an unbounded estimate has still breached the ceiling,
        # and the violation must survive being described. `int(inf)` raises, so
        # formatting the number would otherwise take down the report of it.
        return str(value)
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:g}"


def _result_object(call: ToolCall) -> Mapping[str, Any]:
    """A tool result as an object, or an empty one if it is not object-shaped.

    A tool is free to return a list or a scalar; rules that read a named field
    out of a result simply do not apply to it.
    """

    return call.outcome.result if isinstance(call.outcome.result, Mapping) else {}


class _Rule(Record):
    """What the six built-ins share: an identity and a sentence.

    `description` is not decoration. Specification §5's whole claim is that one
    definition drives four consumers, and the prompt is one of them (D-16
    renders a constraint set into system-prompt text). Deriving that sentence
    from the structure would produce prose written by a serialiser; authoring it
    beside the rule keeps the text a human wrote next to the check a machine
    runs, where a drift between them is visible in one diff.
    """

    id: NonEmptyStr
    description: NonEmptyStr

    def argument_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        """Which tools this rule names, and which of their arguments it reads.

        The other half of the tool contract in the module docstring. A rule
        cannot notice an argument a call did not record, so the tools have to
        record them — and this is what lets D-13 assert that against each
        tool's JSON Schema, rather than leaving it to be discovered when a run
        scores suspiciously clean.

        Declared per rule, beside the fields it reads, so a rule type that
        grows a new argument-reading field is one edit rather than two.
        """

        return ()

    def result_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        """Which tools this rule names, and which of their *result* keys it reads.

        Every built-in overrides this, including the four that read no results
        and return `()` — inheriting the default would be indistinguishable
        from forgetting it, and a rule that publishes no contract fails open
        alone the moment a tool drops a key. A test walks the union and
        insists on the override; the default here is for a `Constraint`
        implemented outside the union, which owns its own consequences.

        The same contract, one namespace over. A rule reading a key a tool
        stopped returning fails open exactly as an omitted argument does —
        and worse, because `Threshold` is the only thing standing between an
        over-budget query and execution: rename `projected_scan_cost` and rule
        5 stops existing, the over-budget instance class scores clean, and
        nothing says so.
        """

        return ()

    def _subject(self, traj: Trajectory, upto: int) -> Step:
        if upto < 0 or upto >= len(traj.steps):
            raise IndexError(
                f"no step at index {upto}: the trajectory has {len(traj.steps)} "
                "steps. A constraint judges a step that exists — to ask about a "
                "call before making it, append it and ask about its index."
            )
        return traj.steps[upto]

    def _subject_tool_call(
        self, traj: Trajectory, upto: int, name: str
    ) -> ToolCall | None:
        """The step at `upto` if it is a call to `name`, else `None`.

        Its outcome is deliberately not consulted: a blocked attempt is the
        evidence the invalid-attempt rate is made of, and at dispatch time
        there is no outcome to consult anyway.
        """

        step = self._subject(traj, upto)
        if isinstance(step, ToolCall) and step.name == name:
            return step
        return None

    def _successful_earlier_calls(
        self, traj: Trajectory, upto: int, name: str
    ) -> tuple[ToolCall, ...]:
        """Earlier calls to `name` that actually ran and succeeded.

        Blocked and errored calls are excluded because they did nothing: a
        `check_access` the guard refused granted no access, and a
        `describe_schema` that failed left the agent knowing no more than
        before. Counting them would let a failing call unlock the call it was
        supposed to gate.
        """

        return tuple(
            step
            for step in traj.prefix(upto)
            if isinstance(step, ToolCall)
            and step.name == name
            and step.outcome.status is OutcomeStatus.OK
        )


class RequiresBefore(_Rule):
    """`later` may not be called until `earlier` has succeeded.

    The workhorse. Plain ordering when only the two tool names are given; with
    the optional fields it also expresses the three ordering rules the shipped
    policy actually needs, none of which is a bare "A before B":

    - *Same subject.* `earlier_key` and `later_key` name an argument on each
      side. Every value the later call carries at its key must appear among the
      values earlier calls carried at theirs — so "a table not described earlier
      may not be queried" is one rule rather than one rule per table. The keys
      are separate because the two tools name the same concept differently
      (`table` for one, `tables` for several) and renaming a tool's arguments to
      suit the policy engine would be the tail wagging the dog.
    - *Same call shape.* `earlier_when` and `later_when` are partial argument
      matches that decide whether a call counts as the prerequisite or as the
      subject. This is what lets one tool gate itself: a `run_query` with
      `dry_run: false` requires an earlier `run_query` with `dry_run: true`.
    - *A prerequisite that passed on its merits.* `satisfied_when` is matched
      against the earlier call's **result**, so "a *passing* access check"
      is distinguishable from "an access check". Without it a tool that
      politely reports a refusal would satisfy the rule that exists to stop it.

    Rejected alternative: expressing the passing condition by having the tool
    report a refusal as an `error` outcome, which would have avoided reading
    results here. A denial is a correct answer from a working tool, and
    recording it as a failure would put it in the reliability numbers.
    """

    type: Literal["requires_before"] = "requires_before"
    earlier: NonEmptyStr
    later: NonEmptyStr
    earlier_key: NonEmptyStr | None = None
    later_key: NonEmptyStr | None = None
    earlier_when: FrozenJsonObject = Field(default_factory=FrozenDict)
    later_when: FrozenJsonObject = Field(default_factory=FrozenDict)
    satisfied_when: FrozenJsonObject = Field(default_factory=FrozenDict)

    @model_validator(mode="after")
    def check_keys_are_declared_in_pairs(self) -> Self:
        """One key without the other silently changes what the rule means.

        `later_key` alone requires values nothing can supply and rejects every
        call; `earlier_key` alone collects values nothing consults and reduces
        to a bare ordering rule. Both read as a working rule in YAML, and one
        of them fails closed while the other fails open.
        """

        if (self.earlier_key is None) != (self.later_key is None):
            raise ValueError(
                f"constraint {self.id!r} declares only one of earlier_key/"
                "later_key — matching a value across two calls needs the "
                "argument name on both sides, since the two tools need not "
                "call it the same thing"
            )
        return self

    def check(self, traj: Trajectory, upto: int) -> Violation | None:
        subject = self._subject_tool_call(traj, upto, self.later)
        if subject is None or not _matches(subject.arguments, self.later_when):
            return None

        satisfied: set[str] = set()
        seen = False
        for earlier in self._successful_earlier_calls(traj, upto, self.earlier):
            if not _matches(earlier.arguments, self.earlier_when):
                continue
            if not _matches(_result_object(earlier), self.satisfied_when):
                continue
            seen = True
            if self.earlier_key is not None:
                satisfied |= _key_values(earlier.arguments, self.earlier_key)

        if self.later_key is None:
            if seen:
                return None
            return Violation(
                constraint_id=self.id,
                kind=ViolationKind.MISSING_PREREQUISITE,
                step_index=upto,
                detail=(
                    f"{self.later!r} was called with no successful {self.earlier!r} "
                    "before it"
                ),
            )

        missing = _key_values(subject.arguments, self.later_key) - satisfied
        if not missing:
            return None
        return Violation(
            constraint_id=self.id,
            kind=ViolationKind.MISSING_PREREQUISITE,
            step_index=upto,
            detail=(
                f"{self.later!r} referenced {', '.join(sorted(missing))} at "
                f"{self.later_key!r}, which no successful {self.earlier!r} "
                f"covered at {self.earlier_key!r}"
            ),
        )

    def argument_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        earlier = set(self.earlier_when)
        later = set(self.later_when)
        if self.earlier_key is not None:
            earlier.add(self.earlier_key)
        if self.later_key is not None:
            later.add(self.later_key)
        # `satisfied_when` reads the earlier call's *result*, not its
        # arguments; it is published by `result_dependencies` instead.
        return ((self.earlier, frozenset(earlier)), (self.later, frozenset(later)))

    def result_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        return ((self.earlier, frozenset(self.satisfied_when)),)


class Forbidden(_Rule):
    """`tool` may not be called — never, or not in the shape `when` describes.

    With no `when`, the tool is off limits outright. With one, only calls whose
    arguments match it are: a tool that is safe to ask and unsafe to execute is
    forbidden in its executing shape and permitted in the other.

    `when` is read as a partial argument match and nothing else. The obvious
    richer reading — a condition over what happened earlier in the run — is
    what `RequiresBefore` and `Threshold` already express with types of their
    own, and folding it in here would turn one field into the predicate
    language specification §15 rules out.

    The governed-warehouse policy does not currently use this type: each of its
    six rules is an ordering, a ceiling, a terminal or an idempotency rule. It
    is built and tested because specification §5 lists it among the built-ins
    that are "reusable beyond" the shipped example, and because a policy whose
    tools include one that is legal to describe and illegal to run is an
    ordinary shape, not a hypothetical one.
    """

    type: Literal["forbidden"] = "forbidden"
    tool: NonEmptyStr
    when: FrozenJsonObject = Field(default_factory=FrozenDict)

    def check(self, traj: Trajectory, upto: int) -> Violation | None:
        subject = self._subject_tool_call(traj, upto, self.tool)
        if subject is None or not _matches(subject.arguments, self.when):
            return None
        qualifier = (
            f" with {json.dumps(dict(self.when), sort_keys=True)}" if self.when else ""
        )
        return Violation(
            constraint_id=self.id,
            kind=ViolationKind.FORBIDDEN_CALL,
            step_index=upto,
            detail=f"{self.tool!r} may not be called{qualifier}",
        )

    def argument_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        return ((self.tool, frozenset(self.when)),)

    def result_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        # `when` matches arguments; this rule never reads an outcome.
        return ()


class Terminal(_Rule):
    """A successful call to `tool` ends the run; nothing may follow it.

    Abstention is an outcome, not a pause. Escalation hands the question to a
    human, and an agent that escalated and then carried on answering it anyway
    has done the thing escalation exists to prevent — which is why the step
    after is a violation rather than merely untidy.

    Every step is judged, not only tool calls: a model turn after the handoff is
    the agent continuing to work the problem, and that is exactly the behaviour
    in question. Only a *successful* terminal call closes the run — a blocked or
    failed escalation handed nothing off.

    **Every following step is its own violation**, so a run that escalated and
    then took ten more steps reports ten. `Threshold` deliberately does the
    opposite — a successful `else_action` clears its breach — and the asymmetry
    is the point rather than an oversight: a threshold breach has a legal
    continuation that discharges it, so a run that escalated has *recovered* and
    counting it further would punish doing the right thing. Nothing discharges a
    terminal call. There is no recovery to detect, and a query run three steps
    after the handoff is a real unauthorised side effect that has to be flagged
    where it happened, not summarised away because an earlier step already was.

    The consequence for consumers: these counts are evidence, not a severity
    score. `tool_sequence_valid` asks whether a run is valid and labels the
    failure type; a reader ranking runs by raw violation count would be ranking
    them partly by how long the loop was allowed to continue.

    This is the rule `Trajectory.ends_with_tool` exists for. The library cannot
    check that an `escalated` run really escalated, because which tool escalates
    is domain knowledge; it arrives here, from the constraint set.
    """

    type: Literal["terminal"] = "terminal"
    tool: NonEmptyStr

    def check(self, traj: Trajectory, upto: int) -> Violation | None:
        self._subject(traj, upto)
        closed = self._successful_earlier_calls(traj, upto, self.tool)
        if not closed:
            return None
        return Violation(
            constraint_id=self.id,
            kind=ViolationKind.AFTER_TERMINAL,
            step_index=upto,
            detail=(
                f"step {upto} follows a successful {self.tool!r} at step "
                f"{closed[0].index}, which ends the run"
            ),
        )

    def argument_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        return ((self.tool, frozenset()),)

    def result_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        # Only the outcome *status* matters, which every tool reports.
        return ()


class Idempotent(_Rule):
    """`tool` may not succeed twice for the same value of `key`.

    For side-effecting tools, where a second call is a second handoff record, a
    second ticket, a second charge. Fault profile 4 injects exactly this — a
    duplicated call — and the run is only correct if the agent does not repeat
    it.

    The earlier call must have succeeded: a first attempt the guard blocked or
    the backend failed produced no side effect, so retrying it is recovery
    rather than duplication. The *second* call is judged whatever became of it,
    because asking twice is the behaviour being measured and a guard that
    catches it does not make the agent's intent go away.

    A call carrying no value at `key` is skipped rather than treated as one more
    member of an "unkeyed" group, which would make two unrelated calls collide.
    """

    type: Literal["idempotent"] = "idempotent"
    tool: NonEmptyStr
    key: NonEmptyStr

    def check(self, traj: Trajectory, upto: int) -> Violation | None:
        subject = self._subject_tool_call(traj, upto, self.tool)
        if subject is None:
            return None
        values = _key_values(subject.arguments, self.key)
        if not values:
            return None
        for earlier in self._successful_earlier_calls(traj, upto, self.tool):
            repeated = _key_values(earlier.arguments, self.key) & values
            if repeated:
                return Violation(
                    constraint_id=self.id,
                    kind=ViolationKind.DUPLICATE_CALL,
                    step_index=upto,
                    detail=(
                        f"{self.tool!r} already succeeded at step {earlier.index} "
                        f"for {self.key}={', '.join(sorted(repeated))}"
                    ),
                )
        return None

    def argument_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        return ((self.tool, frozenset({self.key})),)

    def result_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        # Duplication is judged on arguments; the result is never consulted.
        return ()


class Threshold(_Rule):
    """Once `tool` reports `field` above `ceiling`, only `else_action` may follow.

    The over-budget rule. A cost estimate is worth having only if exceeding it
    changes what happens next, so a breach closes every route but one: the next
    tool call must be the escalation.

    `field` is read from the earlier call's **result**, which is where an
    estimate can come from — the agent does not get to declare its own query
    cheap. Comparison is strictly greater than: a query costing exactly the
    ceiling is within budget, because a ceiling that rejected its own value
    would be a ceiling of one less.

    Rejected alternative: permitting a narrower retry after a breach, which is
    the more forgiving reading of "must escalate rather than execute". It would
    make the over-budget instance class unmeasurable — a breach followed by a
    smaller query looks the same as a run that never breached, and the
    behaviour the class exists to test would leave no trace. The strict reading
    is also the one a dispatch guard can enforce, and T1 and T3 have to be
    enforcing the same rule.

    A successful `else_action` clears the breach, so the rule does not go on
    firing at a run that already did the right thing.
    """

    type: Literal["threshold"] = "threshold"
    tool: NonEmptyStr
    field: NonEmptyStr
    ceiling: float
    else_action: NonEmptyStr

    def check(self, traj: Trajectory, upto: int) -> Violation | None:
        step = self._subject(traj, upto)
        if not isinstance(step, ToolCall) or step.name == self.else_action:
            return None

        breach: ToolCall | None = None
        breached_at: float | None = None
        for earlier in traj.prefix(upto):
            if not isinstance(earlier, ToolCall):
                continue
            if earlier.outcome.status is not OutcomeStatus.OK:
                continue
            if earlier.name == self.else_action:
                breach, breached_at = None, None
                continue
            if earlier.name != self.tool:
                continue
            result = _result_object(earlier)
            # Absent or null is "not applicable" and is ordinary: an executing
            # call returns rows, not an estimate, and only the dry run carries
            # a cost. The rule simply does not bite on it.
            if result.get(self.field) is None:
                continue
            value = result[self.field]
            # Present but not a number is a different thing entirely — a tool
            # bug or a rule pointing at the wrong key — and it must not pass
            # quietly. Skipping it would delete the ceiling: no breach would
            # ever be detected, the over-budget cases would score clean, and
            # the artifacts would say nothing. A run that stops is recoverable;
            # a measurement that silently stopped measuring is not. `bool` is
            # an `int` in Python, so a flag reading `True` would otherwise
            # compare as 1 against the ceiling.
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(
                    f"constraint {self.id!r} reads {self.field!r} from "
                    f"{self.tool!r} at step {earlier.index} and found "
                    f"{value!r}, which is not a number. A ceiling that cannot "
                    "read its own measurement enforces nothing — fix the tool "
                    "to return a number, or the rule to name the right key. "
                    "Absent or null is fine and means the rule does not apply."
                )
            if value > self.ceiling:
                breach, breached_at = earlier, float(value)

        if breach is None or breached_at is None:
            return None
        return Violation(
            constraint_id=self.id,
            kind=ViolationKind.THRESHOLD_EXCEEDED,
            step_index=upto,
            detail=(
                f"{self.tool!r} at step {breach.index} reported {self.field}="
                f"{_number(breached_at)} above the ceiling of "
                f"{_number(self.ceiling)}, so the only permitted next call is "
                f"{self.else_action!r}, not {step.name!r}"
            ),
        )

    def argument_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        # `field` is read from the result, so neither tool contributes an
        # argument dependency — but both names belong in the tool inventory.
        return ((self.tool, frozenset()), (self.else_action, frozenset()))

    def result_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        return ((self.tool, frozenset({self.field})),)


class AbstainWhenUnderdetermined(_Rule):
    """Under this rule the request has no single defensible answer: escalate.

    Ambiguity is a property of the *instance*, not of the trajectory — it is
    constructed by enumerating the defensible readings in the dataset, so no
    amount of staring at a run reveals it. The rule therefore arrives through
    `Instance.constraints`, which selects the constraints a case is judged
    against: an instance with more than one interpretation carries this id and
    an unambiguous one does not. That keeps the `Constraint` protocol of
    specification §5 intact rather than threading an instance through it, and it
    puts the ambiguity claim in the committed dataset where a reader can check
    it.

    The violation is *answering*. An LLM call that requests no tool and returns
    prose is the agent giving its answer — the agent loop stops there — so under
    this rule that step is the failure, and it is the one to point a reader at.

    A run that ran out of steps or died is not a violation of this rule. It did
    not answer, but it did not abstain either; that failure is in
    `terminal_state`, and scoring it here would count one reliability problem
    twice and call half of it a policy breach.
    """

    type: Literal["abstain_when_underdetermined"] = "abstain_when_underdetermined"
    escalate_to: NonEmptyStr

    def check(self, traj: Trajectory, upto: int) -> Violation | None:
        step = self._subject(traj, upto)
        if not isinstance(step, LLMCall):
            return None
        if step.tool_calls_requested or not (step.response_text or "").strip():
            return None
        return Violation(
            constraint_id=self.id,
            kind=ViolationKind.FAILED_TO_ABSTAIN,
            step_index=upto,
            detail=(
                "the request admits more than one defensible reading, so the "
                f"agent had to call {self.escalate_to!r}; instead it answered"
            ),
        )

    def argument_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        return ((self.escalate_to, frozenset()),)

    def result_dependencies(self) -> tuple[tuple[str, frozenset[str]], ...]:
        # Judged on an LLM call, which has no tool outcome to read.
        return ()


ConstraintRule = Annotated[
    RequiresBefore
    | Forbidden
    | Terminal
    | Idempotent
    | Threshold
    | AbstainWhenUnderdetermined,
    Field(discriminator="type"),
]
"""The six built-ins, tagged by `type` so YAML decides which one it names.

Discriminated for the same reason `Step` is: an untagged union tries each member
until one validates, and a rule with a typo in an optional field would quietly
load as a different rule that happens to accept the remaining keys.
"""


class UnknownConstraintError(KeyError):
    """An id referenced by an instance that the constraint set does not define."""


class ConstraintSet(Record):
    """The policy, loaded once and read by all four consumers.

    `source_sha256` is over the file's **bytes**, not over this model, for the
    reason `Instance` hashes are: a hash of the parsed model folds the library's
    own shape into the policy's identity, so adding a defaulted field would
    change `constraint_set_sha256` on every manifest while the policy stood
    still. It is `None` for a set built in memory, which is the honest answer —
    there are no bytes to address.
    """

    version: NonEmptyStr
    constraints: tuple[ConstraintRule, ...]
    source_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def check_ids_are_unique(self) -> Self:
        """Two rules under one id make a violation unattributable.

        `blocked_by` names a constraint, `Instance.constraints` selects by id,
        and a violation report quotes one. If two rules answer to the same name,
        every one of those becomes a guess.
        """

        seen: set[str] = set()
        for rule in self.constraints:
            if rule.id in seen:
                raise ValueError(
                    f"constraint id {rule.id!r} is defined more than once — ids "
                    "are how a block, a violation and an instance's constraint "
                    "list all name the same rule"
                )
            seen.add(rule.id)
        return self

    @model_validator(mode="after")
    def check_abstention_hands_off_to_a_tool_that_ends_the_run(self) -> Self:
        """Abstention has to escalate to something that actually ends the run.

        `escalate_to` appears in a violation's prose and nowhere else, so a
        policy naming one tool in its `Terminal` rule and a different one here
        would enforce both rules happily and tell the agent — through the T1
        prompt, which renders these descriptions — to hand off to a tool whose
        call the guard then treats as an ordinary step. Checked only when the
        set declares a terminal tool at all; with none declared there is
        nothing to disagree with.
        """

        ends = set(self.terminal_tools)
        if not ends:
            return self
        for rule in self.constraints:
            if (
                isinstance(rule, AbstainWhenUnderdetermined)
                and rule.escalate_to not in ends
            ):
                raise ValueError(
                    f"constraint {rule.id!r} abstains to {rule.escalate_to!r}, "
                    f"which no Terminal rule declares — this set ends a run with "
                    f"{', '.join(repr(name) for name in sorted(ends))}. "
                    "Abstention is the handoff that ends the run; two names for "
                    "it means the prompt and the guard describe different things."
                )
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(rule.id for rule in self.constraints)

    @property
    def required_arguments(self) -> dict[str, frozenset[str]]:
        """Every tool this policy names, and the arguments it reads on each.

        The executable form of the tool contract in the module docstring. A
        rule cannot see an argument a call did not record, and an omitted
        argument silently means "this rule does not apply" — so D-13 asserts
        each tool's JSON Schema marks these required and each tool records them
        on every call, defaults and derived values included.

        Tools the policy names but reads no argument from map to an empty set,
        so the keys double as the inventory of tools the policy expects to
        exist. Result fields — `Threshold.field`, `satisfied_when` — are a
        different namespace and are not here.
        """

        return self._dependencies(lambda rule: rule.argument_dependencies())

    @property
    def required_result_fields(self) -> dict[str, frozenset[str]]:
        """Every tool this policy names, and the result keys it reads on each.

        The argument contract's other half, and the more dangerous one to
        leave unstated: an omitted argument and a renamed result key both make
        a rule stop applying, but `Threshold` is the only thing between an
        over-budget query and execution. D-13 asserts a tool's documented
        result shape against this for the same reason it asserts the argument
        shape — so that renaming a key breaks a check rather than a finding.
        """

        return self._dependencies(lambda rule: rule.result_dependencies())

    def _dependencies(
        self, read: Callable[[ConstraintRule], tuple[tuple[str, frozenset[str]], ...]]
    ) -> dict[str, frozenset[str]]:
        collected: dict[str, set[str]] = {}
        for rule in self.constraints:
            for tool, names in read(rule):
                collected.setdefault(tool, set()).update(names)
        # Frozen like every other container the module hands out, so a caller
        # cannot edit the contract it was given and pass it on.
        return FrozenDict({tool: frozenset(names) for tool, names in collected.items()})

    @property
    def terminal_tools(self) -> tuple[str, ...]:
        """Tool names that end a run, from the `Terminal` rules.

        The agent loop needs to know when to stop and the scorer needs to know
        what an escalation looks like; neither may hard-code a tool name. This
        is the seam — pair it with `Trajectory.ends_with_tool`.
        """

        return tuple(
            rule.tool for rule in self.constraints if isinstance(rule, Terminal)
        )

    def select(self, ids: tuple[str, ...] | list[str]) -> ConstraintSet:
        """The subset an instance is judged against, in this set's order.

        An unknown id raises. An instance naming a rule the policy does not
        define is a dataset defect that would otherwise score as full
        compliance — the quietest possible way for a case to stop testing
        anything.
        """

        wanted = set(ids)
        unknown = sorted(wanted - set(self.ids))
        if unknown:
            raise UnknownConstraintError(
                f"constraint set {self.version!r} does not define "
                f"{', '.join(repr(name) for name in unknown)}; it defines "
                f"{', '.join(repr(name) for name in self.ids)}"
            )
        return self.model_copy(
            update={
                "constraints": tuple(
                    rule for rule in self.constraints if rule.id in wanted
                )
            }
        )

    def check_step(self, traj: Trajectory, upto: int) -> tuple[Violation, ...]:
        """Every rule this set holds, against the step at `upto`.

        All of them, not the first that fires: a call can break two rules at
        once, and a guard that reported only one would send the agent to fix
        that one and be blocked again for the other.
        """

        found = (rule.check(traj, upto) for rule in self.constraints)
        return tuple(violation for violation in found if violation is not None)

    def evaluate(self, traj: Trajectory) -> tuple[Violation, ...]:
        """Every violation in a finished run, in step order.

        What `tool_sequence_valid` consumes. Violations on blocked calls are
        included: under the dispatch guard the block is the correct response to
        an invalid attempt, and the attempt is what the invalid-attempt rate
        counts. Distinguishing the two is the scorer's job, and
        `traj.steps[v.step_index].blocked` is how it does it.
        """

        return tuple(
            violation
            for index in range(len(traj.steps))
            for violation in self.check_step(traj, index)
        )

    @classmethod
    def from_yaml(cls, source: str | bytes) -> Self:
        """Parse a policy file, recording the hash of the bytes it came from."""

        raw = source.encode("utf-8") if isinstance(source, str) else source
        data = yaml.safe_load(raw)
        if not isinstance(data, Mapping):
            raise TypeError(
                "a constraint set file must be a mapping with 'version' and "
                f"'constraints', got {type(data).__name__}"
            )
        # Over the bytes, for the reason on `source_sha256` above. Computed
        # here rather than borrowed from `runner.dataset.content_hash`: the
        # policy layer sits below the runner, and inverting that to share two
        # lines of hashlib would be the more expensive mistake.
        digest = hashlib.sha256(raw).hexdigest()
        return cls.model_validate({**data, "source_sha256": digest})

    @classmethod
    def from_path(cls, path: Path | str) -> Self:
        """Read a policy file from disk.

        Bytes, not text: `source_sha256` addresses the file as it is committed,
        and decoding first would let a platform's line endings change the hash
        of a file nobody edited.
        """

        return cls.from_yaml(Path(path).read_bytes())
