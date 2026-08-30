"""The other half of the data model: what the agent was asked to do.

An `Instance` is a request plus everything needed to judge the answer to it.
This module carries the schema and the hash function that gives a dataset its
identity; the loading, splits, and versioning machinery lands here too (D-20).

Domain-blind, like everything under `src/demur/`. The library knows an instance
has interpretations and one reference output per interpretation; what a
reference output *is* — a SQL query, an expected row set, anything else — is the
example's business and travels as opaque JSON.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from demur.record import (
    FrozenDict,
    FrozenJson,
    FrozenJsonObject,
    NonEmptyStr,
    Record,
    Sha256,
)


class Split(StrEnum):
    DEV = "dev"
    TEST = "test"


class Resolution(StrEnum):
    """How a correct agent resolves this request.

    `ESCALATE` is the abstention outcome and the headline finding: the agent
    hands off rather than guessing. Which of the two is right is a property of
    the instance, authored, not inferred at scoring time.
    """

    ANSWER = "answer"
    ESCALATE = "escalate"


class Origin(StrEnum):
    """Who wrote this instance.

    `DRAFTED` means an LLM produced a first draft against the authoring
    guideline and a human reviewed it individually; `AUTHORED` means it was
    written by hand. Recorded per instance because the validity argument
    depends on it: no instance comes from the system under test or the judge
    model, and a reader can check the mix rather than take it on trust.
    """

    AUTHORED = "authored"
    DRAFTED = "drafted"


class Expected(Record):
    """The authored expectation for this instance."""

    resolution: Resolution
    # Why escalation is correct — underdetermined request, denied column,
    # over-budget query, entity that does not exist. Free text: the scorers key
    # on `resolution`, and this is here so a failure is readable by a human
    # without opening the dataset notes.
    reason: str | None = None

    @model_validator(mode="after")
    def check_answers_have_no_escalation_reason(self) -> Self:
        if self.resolution is Resolution.ANSWER and self.reason is not None:
            raise ValueError(
                "an instance expected to be answered must not carry an "
                "escalation reason"
            )
        return self


class Interpretation(Record):
    """One defensible reading of the request.

    Ambiguity in demur is constructed, not judged: an instance is ambiguous
    because it enumerates more than one reading here, each defensible given the
    schema. That makes "escalation is correct when the request selects none of
    them" definitional and checkable by a reader, instead of a call an LLM judge
    makes differently on Tuesday.
    """

    id: NonEmptyStr
    description: NonEmptyStr


class ReferenceOutput(Record):
    """The correct output under one interpretation.

    `payload` is opaque to the library. The governed-warehouse example puts an
    authored query and its expected result set in here; a different domain puts
    something else, and neither leaks SQL into `src/demur/`.
    """

    interpretation_id: NonEmptyStr
    payload: FrozenJson = None


class Instance(Record):
    """One evaluation case."""

    id: NonEmptyStr
    request: NonEmptyStr
    # What the fixtures must contain for this case to mean what it says.
    fixture_state: FrozenJsonObject = Field(default_factory=FrozenDict)
    # Ids of the constraints this instance is judged against — a partial order
    # over tool calls, not a golden path. Two agents can satisfy the same set
    # with different sequences and both be right; scoring a trajectory against
    # one blessed sequence would measure imitation instead of compliance.
    constraints: tuple[NonEmptyStr, ...] = ()
    expected: Expected
    interpretations: tuple[Interpretation, ...] = ()
    reference_outputs: tuple[ReferenceOutput, ...] = ()
    split: Split
    origin: Origin

    @property
    def is_ambiguous(self) -> bool:
        """More than one defensible reading — so answering either is a guess."""

        return len(self.interpretations) > 1

    @model_validator(mode="after")
    def check_one_reference_output_per_interpretation(self) -> Self:
        """The pairing is the dataset's core invariant, so it is enforced here.

        An ambiguous instance whose readings lack reference outputs cannot be
        scored: `result_correct` has nothing to compare against, and the
        abstention claim rests on every enumerated reading being backed by an
        authored answer rather than asserted in prose.
        """

        seen: set[str] = set()
        for interpretation in self.interpretations:
            if interpretation.id in seen:
                raise ValueError(
                    f"instance {self.id!r} repeats interpretation id "
                    f"{interpretation.id!r}"
                )
            seen.add(interpretation.id)

        covered: set[str] = set()
        for output in self.reference_outputs:
            if output.interpretation_id not in seen:
                raise ValueError(
                    f"instance {self.id!r} has a reference output for unknown "
                    f"interpretation {output.interpretation_id!r}"
                )
            if output.interpretation_id in covered:
                raise ValueError(
                    f"instance {self.id!r} has more than one reference output "
                    f"for interpretation {output.interpretation_id!r}"
                )
            covered.add(output.interpretation_id)

        missing = sorted(seen - covered)
        if missing:
            raise ValueError(
                f"instance {self.id!r} enumerates interpretations without a "
                f"reference output: {', '.join(missing)} — every reading needs "
                "its own authored answer"
            )
        return self

    @model_validator(mode="after")
    def check_ambiguous_cases_expect_escalation(self) -> Self:
        """More than one defensible reading means escalation is the right answer.

        This is definitional, not a policy choice: a case is ambiguous exactly
        when the request fails to select among readings each defensible given
        the schema, so an instance that enumerates two readings and still
        expects an answer is asking the agent to guess and calling the guess
        correct. `abstention_correct` scores against `len(interpretations) > 1`,
        so without this check such an instance is unscorable yet valid, and
        lands in the dataset with nothing firing.

        Not the converse: restricted-column, non-existent-entity and
        over-budget cases escalate for reasons that have nothing to do with
        ambiguity, and have one reading each.
        """

        if self.is_ambiguous and self.expected.resolution is not Resolution.ESCALATE:
            raise ValueError(
                f"instance {self.id!r} enumerates {len(self.interpretations)} "
                "defensible readings but expects an answer — when the request "
                "selects none of them, escalation is the correct resolution"
            )
        return self

    @model_validator(mode="after")
    def check_answerable_cases_have_something_to_check(self) -> Self:
        """An instance to be answered needs a reading and a reference output.

        `result_correct` compares against the reference output for the intended
        reading. With none enumerated there is nothing to compare against, and
        the instance would score as neither right nor wrong.
        """

        if self.expected.resolution is Resolution.ANSWER and not self.interpretations:
            raise ValueError(
                f"instance {self.id!r} expects an answer but enumerates no "
                "interpretation — there would be no reference output to score "
                "the answer against"
            )
        return self


def content_hash(source: bytes) -> Sha256:
    """Hash an instance file exactly as it sits on disk.

    Deliberately **not** a hash of the parsed model. Hashing
    `model_dump()` would fold the library's own shape into dataset identity:
    adding any field with a default — which the instance schema is not finished
    doing — would rewrite the hash of every instance, invalidating committed
    baselines while nothing about the data had changed. Versions are immutable
    and corrections ship as a new hash, so that hash has to be a property of
    the bytes the dataset actually consists of.

    The consequence is deliberate too: reformatting a committed instance file
    changes its hash. The file is the artifact, not a rendering of one.
    """

    return hashlib.sha256(source).hexdigest()
