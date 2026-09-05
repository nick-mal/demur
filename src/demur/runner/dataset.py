"""What the agent was asked to do. See spec §4, Instance.

Domain-blind: a reference output is opaque JSON, so a SQL query or an expected
row set never enters the library. Loading, splits and versioning land here.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from demur.policy.constraints import ConstraintSet
from demur.record import FrozenDict, FrozenJson, FrozenJsonObject, NonEmptyStr, Record


class Split(StrEnum):
    """Which half of the dataset an instance belongs to."""

    DEV = "dev"
    TEST = "test"


class Resolution(StrEnum):
    """How a correct agent resolves a request. Authored, never inferred."""

    ANSWER = "answer"
    ESCALATE = "escalate"


class Origin(StrEnum):
    """Who wrote an instance.

    Recorded because the validity argument depends on it: nothing comes from
    the system under test or the judge model, and a reader can check the mix.
    """

    AUTHORED = "authored"
    DRAFTED = "drafted"


class Expected(Record):
    """The authored expectation for an instance."""

    resolution: Resolution
    # Why escalation is correct. Free text for a human reader; scorers key on
    # `resolution`.
    reason: str | None = None

    @model_validator(mode="after")
    def check_answers_have_no_escalation_reason(self) -> Self:
        """Reject an escalation reason on an instance expected to be answered."""

        if self.resolution is Resolution.ANSWER and self.reason is not None:
            raise ValueError(
                "an instance expected to be answered must not carry an "
                "escalation reason"
            )
        return self


class Interpretation(Record):
    """One defensible reading of a request, with its authored answer.

    Ambiguity is constructed by enumerating these, not judged. Escalation is
    correct when the request selects none of them, which a reader can check.
    The answer lives here so a reading cannot exist without one.
    """

    id: NonEmptyStr
    description: NonEmptyStr
    # The correct output under this reading. Opaque to the library: the
    # warehouse example puts a query and its rows here.
    reference_output: FrozenJson = None


class Instance(Record):
    """One evaluation case: a request and everything needed to judge the answer."""

    id: NonEmptyStr
    request: NonEmptyStr
    # What the fixtures must contain for this case to mean what it says.
    fixture_state: FrozenJsonObject = Field(default_factory=FrozenDict)
    # Ids of the constraints this instance is judged against: a partial order,
    # not a golden path. Scoring against one blessed sequence measures imitation.
    constraints: tuple[NonEmptyStr, ...] = ()
    expected: Expected
    interpretations: tuple[Interpretation, ...] = ()
    split: Split
    origin: Origin

    @property
    def is_ambiguous(self) -> bool:
        """Return whether the request has more than one defensible reading.

        Answering either of them is then a guess.
        """

        return len(self.interpretations) > 1

    @model_validator(mode="after")
    def check_interpretation_ids_are_unique(self) -> Self:
        """Reject two interpretations sharing an id."""

        seen: set[str] = set()
        for interpretation in self.interpretations:
            if interpretation.id in seen:
                raise ValueError(
                    f"instance {self.id!r} repeats interpretation id "
                    f"{interpretation.id!r}"
                )
            seen.add(interpretation.id)
        return self

    @model_validator(mode="after")
    def check_ambiguous_cases_expect_escalation(self) -> Self:
        """Reject an ambiguous instance that expects an answer.

        Definitional: two readings and an expected answer asks the agent to
        guess and calls the guess correct. Not the converse: a restricted
        column or an over-budget query escalates with one reading.
        """

        if self.is_ambiguous and self.expected.resolution is not Resolution.ESCALATE:
            raise ValueError(
                f"instance {self.id!r} enumerates {len(self.interpretations)} "
                "defensible readings but expects an answer. When the request "
                "selects none of them, escalation is the correct resolution."
            )
        return self

    @model_validator(mode="after")
    def check_answerable_cases_have_something_to_check(self) -> Self:
        """Reject an answerable instance with no interpretation.

        `result_correct` compares against the reference output of the intended
        reading. With none there is nothing to compare against.
        """

        if self.expected.resolution is Resolution.ANSWER and not self.interpretations:
            raise ValueError(
                f"instance {self.id!r} expects an answer but enumerates no "
                "interpretation, so there is no reference output to score "
                "the answer against."
            )
        return self


def check_constraints_cover_ambiguity(
    instance: Instance, policy: ConstraintSet
) -> None:
    """Raise `ValueError` if an ambiguous instance selects no abstention rule.

    `ConstraintSet.select` catches the loud direction, an unknown id. This is
    the quiet one: an ambiguous case that forgets the rule passes every
    constraint it did select and scores as full compliance while testing
    nothing. Raises rather than reporting: an unscorable dataset is not one to
    run against.
    """

    if not instance.is_ambiguous:
        return

    selected = set(instance.constraints)
    available = policy.abstention_ids
    if any(name in selected for name in available):
        return

    listed = ", ".join(repr(name) for name in instance.constraints) or "(none)"
    remedy = (
        f"Add {' or '.join(repr(name) for name in available)} to them."
        if available
        else (
            f"Constraint set {policy.version!r} defines no abstention rule at "
            "all, so it cannot judge an ambiguous case."
        )
    )
    raise ValueError(
        f"instance {instance.id!r} enumerates {len(instance.interpretations)} "
        "defensible readings but selects no abstention rule, so nothing would "
        f"check that it escalated: its constraints are {listed}. {remedy}"
    )
