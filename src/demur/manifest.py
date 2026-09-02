"""Per-run identity and provenance.

Written once per run, and holding everything that stays constant across it. A
trajectory says what happened in one attempt at one instance; the manifest says
what the run *was* — which model, which treatment, which dataset, which prompt.
Keeping that off the trajectory means a run's identity is recorded once rather
than copied onto every artifact it produced, where the copies could disagree.

Two of the fields carry the experimental control. `system_prompt_sha256` being
byte-identical across the T1, T2 and T3 manifests is the evidence that no
policy text was dropped from a treatment — the comparison rests on all three
carrying the same policy prompt, and a hash proves it where a claim in a
methodology document only asserts it. `constraint_set_sha256` does the same for
the rules the guard and the scorer both read.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from demur.record import FrozenDict, FrozenStrMap, NonEmptyStr, Record, Sha256
from demur.sampling import Sampling


class Treatment(StrEnum):
    """Enforcement placement, strictly incremental.

    T2 and T3 each add one thing to T1 and change nothing else. All three carry
    the identical T1 policy prompt: stripping it from T3 would measure
    information removal rather than where enforcement sits.
    """

    T1_PROMPT = "T1"
    T2_FEW_SHOT = "T2"
    T3_DISPATCH_GUARD = "T3"


class RunManifest(Record):
    """What was held constant for one run."""

    run_id: NonEmptyStr
    started_at: datetime

    # What was evaluated.
    dataset_version: NonEmptyStr
    dataset_sha256: Sha256
    # Every scorer that ran, `id` → `version`. A version change invalidates
    # dependent baselines, so the run has to record which ones it used.
    scorer_versions: FrozenStrMap = Field(default_factory=FrozenDict)

    # What did the evaluating.
    model: NonEmptyStr
    provider: NonEmptyStr
    treatment: Treatment
    prompt_version: NonEmptyStr
    system_prompt_sha256: Sha256
    constraint_set_sha256: Sha256
    demur_version: NonEmptyStr
    # The decoding parameters the run was configured with. Required, and
    # required to pin a temperature: every finding demur publishes is a claim
    # about behaviour at a configuration, and a baseline sampled greedily
    # against a candidate sampled at temperature 1 compares two systems. This
    # belongs in the baseline content-address alongside `model` for exactly
    # that reason. What each call actually sent is `LLMCall.sampling`.
    sampling: Sampling

    # How it was run.
    repeats: int = Field(default=1, ge=1)
    # The **harness** seed: instance ordering, fault-profile selection, any
    # stochastic scorer. Not the model's — that is `sampling.seed`, a different
    # RNG entirely, and conflating the two is how a run looks reproducible on
    # paper while the model still wanders. `None` when the run was not seeded,
    # which is not the same as seed 0 and is the difference between "not
    # reproducible" and "reproducible from here".
    seed: int | None = None
    # Absent until tolerance bands exist; recorded from then on, so that a band
    # edited after a candidate ran is visible as a changed hash.
    tolerances_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def check_the_temperature_was_pinned(self) -> Self:
        """A run at "whatever the provider defaults to" is not a measurement.

        Provider defaults change under stable model aliases, so a run recorded
        without a temperature cannot be reproduced even in principle, and a
        later run that differs cannot be attributed: the candidate changed, or
        the default did, and the artifacts do not say which. `0` is a fine
        answer and the usual one. Not stating one is not.
        """

        if self.sampling.temperature is None:
            raise ValueError(
                f"run {self.run_id!r} records no temperature. `None` means the "
                "provider chose, and provider defaults move under stable model "
                "aliases — so the run is unreproducible and a later difference "
                "cannot be attributed to the candidate rather than to the "
                "default. Record the value the adapter actually sent; 0 is a "
                "fine answer."
            )
        return self

    @model_validator(mode="after")
    def check_started_at_is_absolute(self) -> Self:
        """A naive timestamp is only meaningful on the machine that wrote it.

        Runs get compared across machines and across months, so the offset has
        to be in the record rather than inferred from where it is read.
        """

        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ValueError(
                f"started_at ({self.started_at.isoformat()}) has no timezone — "
                "record an absolute instant, not a local wall-clock reading"
            )
        return self
