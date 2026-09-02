"""Policy as data: ordering and permission rules evaluated into violations.

One constraint set drives the T1 prompt, the T2 exemplars, the T3 guard and
the `tool_sequence_valid` scorer, so the treatments enforce the same rules.
Domain-blind and not a DSL: every field is a tool name, an argument name, a
literal or a number. See spec §5.

Evaluation is per step: `check(traj, upto)` judges the step at `upto` given
everything before it. The subject is judged whatever became of it; an earlier
step counts only if it succeeded. Rules fail open on a missing argument, so
the tools must record every argument the policy reads. See spec §11.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import Field, model_validator

from demur.record import (
    FrozenDict,
    FrozenJsonObject,
    NonEmptyStr,
    Record,
    Sha256,
    sha256_hex,
)
from demur.trajectory import LLMCall, OutcomeStatus, Step, ToolCall, Trajectory


class ViolationKind(StrEnum):
    """One kind per constraint type, not per rule, so the failure-category
    confusion table keeps fixed columns while a policy renames rules."""

    MISSING_PREREQUISITE = "missing_prerequisite"
    FORBIDDEN_CALL = "forbidden_call"
    AFTER_TERMINAL = "after_terminal"
    DUPLICATE_CALL = "duplicate_call"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    FAILED_TO_ABSTAIN = "failed_to_abstain"


class Violation(Record):
    """One rule broken at one step.

    No `blocked` field: `traj.steps[step_index].blocked` already says so, and
    storing it twice lets two readers disagree. `detail` names the values
    involved, for a human.
    """

    constraint_id: NonEmptyStr
    kind: ViolationKind
    step_index: int = Field(ge=0)
    detail: NonEmptyStr


Dependencies = tuple[tuple[str, frozenset[str]], ...]
"""Tool names paired with the argument or result keys a rule reads on them."""


def _literal(value: Any) -> str:
    """A JSON value as a type-distinguishing string, so `1` and `"1"` differ."""

    return json.dumps(value, sort_keys=True)


def _key_values(source: Mapping[str, Any], key: str) -> frozenset[str]:
    """The value or values at `key`, as a set. A scalar and a list both become
    sets, so one rule can relate `table` to `tables`. A missing key yields the
    empty set: it satisfies nothing and requires nothing."""

    if key not in source:
        return frozenset()
    value = source[key]
    items = value if isinstance(value, list | tuple) else [value]
    return frozenset(_literal(item) for item in items)


def _matches(source: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Partial match: every key in `expected` is present in `source` with an
    equal literal. A missing key never matches, so the rule is skipped."""

    return all(
        key in source and _literal(source[key]) == _literal(value)
        for key, value in expected.items()
    )


def _number(value: float) -> str:
    """A measurement as a reader would write it, never in exponent notation."""

    if not math.isfinite(value):
        return str(value)
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:g}"


def _result_object(call: ToolCall) -> Mapping[str, Any]:
    """A tool result as an object, or empty if it is not object-shaped."""

    return call.outcome.result if isinstance(call.outcome.result, Mapping) else {}


class _Rule(Record):
    """What the six built-ins share: an id and a sentence.

    `description` is rendered into the T1 prompt, so it is authored beside the
    check it describes and a drift between them shows in one diff. Subclasses
    publish the arguments and result keys they read. There is no default: a
    type that forgets fails at first use rather than failing open.
    """

    id: NonEmptyStr
    description: NonEmptyStr

    def _subject(self, traj: Trajectory, upto: int) -> Step:
        if upto < 0 or upto >= len(traj.steps):
            raise IndexError(
                f"no step at index {upto}: the trajectory has {len(traj.steps)} "
                "steps. To ask about a call before making it, append it and ask "
                "about its index."
            )
        return traj.steps[upto]

    def _subject_tool_call(
        self, traj: Trajectory, upto: int, name: str
    ) -> ToolCall | None:
        """The step at `upto` if it calls `name`, else `None`. Its outcome is
        not consulted: a blocked attempt is still an attempt."""

        step = self._subject(traj, upto)
        if isinstance(step, ToolCall) and step.name == name:
            return step
        return None

    def _successful_earlier_calls(
        self, traj: Trajectory, upto: int, name: str
    ) -> tuple[ToolCall, ...]:
        """Earlier calls to `name` that ran and succeeded. A blocked or failed
        call did nothing, so it unlocks nothing."""

        return tuple(
            step
            for step in traj.steps[:upto]
            if isinstance(step, ToolCall)
            and step.name == name
            and step.outcome.status is OutcomeStatus.OK
        )


class RequiresBefore(_Rule):
    """`later` may not be called until `earlier` has succeeded.

    `earlier_key`/`later_key` relate values across the two calls, so one rule
    covers every table. `earlier_when`/`later_when` pick which calls count, so
    a tool can gate itself. `satisfied_when` matches the earlier call's result,
    so a denial is not a passing check. A denial is a correct answer from a
    working tool, so it is not recorded as an error either.
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
        """`later_key` alone rejects every call; `earlier_key` alone reduces
        to a bare ordering rule. Both read as a working rule in YAML."""

        if (self.earlier_key is None) != (self.later_key is None):
            raise ValueError(
                f"constraint {self.id!r} declares only one of earlier_key/"
                "later_key. Matching a value across two calls needs the argument "
                "name on both sides."
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

    def argument_dependencies(self) -> Dependencies:
        earlier = set(self.earlier_when)
        later = set(self.later_when)
        if self.earlier_key is not None:
            earlier.add(self.earlier_key)
        if self.later_key is not None:
            later.add(self.later_key)
        return ((self.earlier, frozenset(earlier)), (self.later, frozenset(later)))

    def result_dependencies(self) -> Dependencies:
        return ((self.earlier, frozenset(self.satisfied_when)),)


class Forbidden(_Rule):
    """`tool` may not be called: never, or not in the shape `when` describes.

    `when` is a partial argument match and nothing more. A condition over
    earlier steps is what `RequiresBefore` and `Threshold` are for; folding it
    in here would make one field the predicate language spec §15 rules out.
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

    def argument_dependencies(self) -> Dependencies:
        return ((self.tool, frozenset(self.when)),)

    def result_dependencies(self) -> Dependencies:
        return ()


class Terminal(_Rule):
    """A successful call to `tool` ends the run; nothing may follow it.

    Every later step is its own violation, model turns included: a query run
    three steps after the handoff is a real side effect and is flagged where
    it happened. `Threshold` does the opposite because a breach has a legal
    continuation that discharges it; nothing discharges a terminal call. So
    violation counts are evidence, not a severity score.
    """

    type: Literal["terminal"] = "terminal"
    tool: NonEmptyStr

    def check(self, traj: Trajectory, upto: int) -> Violation | None:
        # Bounds check only: every kind of step after the handoff is judged.
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

    def argument_dependencies(self) -> Dependencies:
        return ((self.tool, frozenset()),)

    def result_dependencies(self) -> Dependencies:
        return ()


class Idempotent(_Rule):
    """`tool` may not succeed twice for the same value of `key`.

    A first attempt that was blocked or failed produced no side effect, so
    retrying it is recovery. A call with no value at `key` is skipped rather
    than grouped as "unkeyed", which would make unrelated calls collide.
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

    def argument_dependencies(self) -> Dependencies:
        return ((self.tool, frozenset({self.key})),)

    def result_dependencies(self) -> Dependencies:
        return ()


class Threshold(_Rule):
    """Once `tool` reports `field` above `ceiling`, only `else_action` may follow.

    Read from the result, so the agent cannot declare its own query cheap.
    Strictly greater than. A narrower retry after a breach is rejected: it
    would look like a run that never breached, and the over-budget class would
    be unmeasurable. A successful `else_action` clears the breach.
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

        breach: tuple[ToolCall, float] | None = None
        for earlier in traj.steps[:upto]:
            if not isinstance(earlier, ToolCall):
                continue
            if earlier.outcome.status is not OutcomeStatus.OK:
                continue
            if earlier.name == self.else_action:
                breach = None
                continue
            if earlier.name != self.tool:
                continue
            value = _result_object(earlier).get(self.field)
            # Absent or null means the rule does not apply: only a dry run
            # carries an estimate.
            if value is None:
                continue
            # Present but not a number is a tool bug or a rule naming the wrong
            # key. Skipping it would delete the ceiling silently. `bool` is an
            # `int` in Python, so a flag would otherwise compare as 1.
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(
                    f"constraint {self.id!r} reads {self.field!r} from "
                    f"{self.tool!r} at step {earlier.index} and found {value!r}, "
                    "which is not a number. Fix the tool to return a number, or "
                    "the rule to name the right key."
                )
            if value > self.ceiling:
                breach = (earlier, float(value))

        if breach is None:
            return None
        source, measured = breach
        return Violation(
            constraint_id=self.id,
            kind=ViolationKind.THRESHOLD_EXCEEDED,
            step_index=upto,
            detail=(
                f"{self.tool!r} at step {source.index} reported {self.field}="
                f"{_number(measured)} above the ceiling of "
                f"{_number(self.ceiling)}, so the only permitted next call is "
                f"{self.else_action!r}, not {step.name!r}"
            ),
        )

    def argument_dependencies(self) -> Dependencies:
        # Both tools belong in the inventory; neither contributes an argument.
        return ((self.tool, frozenset()), (self.else_action, frozenset()))

    def result_dependencies(self) -> Dependencies:
        return ((self.tool, frozenset({self.field})),)


class AbstainWhenUnderdetermined(_Rule):
    """The request has no single defensible answer, so the run must escalate.

    Ambiguity is a property of the instance, so this rule arrives through
    `Instance.constraints` rather than by threading an instance through
    `check`. The violation is answering: an LLM call with prose and no tool
    call. A run that died without answering is a `terminal_state` problem,
    not a breach of this rule.
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

    def argument_dependencies(self) -> Dependencies:
        return ((self.escalate_to, frozenset()),)

    def result_dependencies(self) -> Dependencies:
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
"""The six built-ins, tagged by `type`. Untagged, a rule with a typo in an
optional field would load as whichever member accepts the remaining keys."""


class UnknownConstraintError(KeyError):
    """An id referenced by an instance that the constraint set does not define."""


class ConstraintSet(Record):
    """The policy, loaded once and read by all four consumers.

    `source_sha256` is over the file's bytes, not this model, so a library
    change cannot move `constraint_set_sha256` while the policy stands still.
    `None` for a set built in memory or selected from a larger one.
    """

    version: NonEmptyStr
    constraints: tuple[ConstraintRule, ...]
    source_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def check_ids_are_unique(self) -> Self:
        """`blocked_by`, a violation and an instance's constraint list all name
        a rule by id. Two rules under one id make each of those a guess."""

        seen: set[str] = set()
        for rule in self.constraints:
            if rule.id in seen:
                raise ValueError(
                    f"constraint id {rule.id!r} is defined more than once. Ids "
                    "are how a block, a violation and an instance name a rule."
                )
            seen.add(rule.id)
        return self

    @model_validator(mode="after")
    def check_abstention_hands_off_to_a_tool_that_ends_the_run(self) -> Self:
        """`escalate_to` is rendered into the prompt. Naming a tool no
        `Terminal` rule declares tells the agent to hand off to a call the
        guard treats as an ordinary step."""

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
                    f"which no Terminal rule declares; this set ends a run with "
                    f"{', '.join(repr(name) for name in sorted(ends))}. The prompt "
                    "and the guard must name the same handoff."
                )
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(rule.id for rule in self.constraints)

    @property
    def required_arguments(self) -> Mapping[str, frozenset[str]]:
        """Every tool this policy names, and the arguments it reads on each.

        The tool contract of spec §11, in executable form. Tools with no
        arguments read map to an empty set, so the keys are the tool inventory.
        """

        return self._dependencies(lambda rule: rule.argument_dependencies())

    @property
    def required_result_fields(self) -> Mapping[str, frozenset[str]]:
        """Every tool this policy names, and the result keys it reads on each.

        The more dangerous half: a renamed result key under `Threshold` means
        no breach is ever detected and the over-budget cases score clean.
        """

        return self._dependencies(lambda rule: rule.result_dependencies())

    def _dependencies(
        self, read: Callable[[ConstraintRule], Dependencies]
    ) -> Mapping[str, frozenset[str]]:
        collected: dict[str, set[str]] = {}
        for rule in self.constraints:
            for tool, names in read(rule):
                collected.setdefault(tool, set()).update(names)
        # Frozen, so a caller cannot edit the contract and pass it on.
        return FrozenDict({tool: frozenset(names) for tool, names in collected.items()})

    @property
    def terminal_tools(self) -> tuple[str, ...]:
        """Tool names that end a run. The loop and the scorer read this rather
        than hard-coding a name; pair with `Trajectory.ends_with_tool`."""

        return tuple(
            rule.tool for rule in self.constraints if isinstance(rule, Terminal)
        )

    def select(self, ids: Iterable[str]) -> ConstraintSet:
        """The subset an instance is judged against, in this set's order. An
        unknown id raises: it would otherwise score as full compliance."""

        wanted = set(ids)
        unknown = sorted(wanted - set(self.ids))
        if unknown:
            raise UnknownConstraintError(
                f"constraint set {self.version!r} does not define "
                f"{', '.join(repr(name) for name in unknown)}; it defines "
                f"{', '.join(repr(name) for name in self.ids)}"
            )
        # No hash: the subset is not the file. The manifest records the full set's.
        return self.model_copy(
            update={
                "constraints": tuple(
                    rule for rule in self.constraints if rule.id in wanted
                ),
                "source_sha256": None,
            }
        )

    def check_step(self, traj: Trajectory, upto: int) -> tuple[Violation, ...]:
        """Every rule against the step at `upto`. All of them, not the first:
        a guard reporting one would send the agent to fix it and block again."""

        found = (rule.check(traj, upto) for rule in self.constraints)
        return tuple(violation for violation in found if violation is not None)

    def evaluate(self, traj: Trajectory) -> tuple[Violation, ...]:
        """Every violation in a finished run, in step order. Violations on
        blocked calls are included; `traj.steps[i].blocked` tells them apart."""

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
        return cls.model_validate({**data, "source_sha256": sha256_hex(raw)})

    @classmethod
    def from_path(cls, path: Path | str) -> Self:
        """Read a policy file as bytes, so line endings cannot change the hash."""

        return cls.from_yaml(Path(path).read_bytes())
