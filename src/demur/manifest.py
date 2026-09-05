"""Per-run identity and provenance. See spec §4, RunManifest.

A trajectory says what happened in one attempt; the manifest says what the
run was. Recorded once per run, so copies cannot disagree.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from demur.record import FrozenDict, FrozenStrMap, NonEmptyStr, Record, Sha256
from demur.sampling import Sampling


class Treatment(StrEnum):
    """Where enforcement sits for a run. T2 and T3 each add one thing to T1.

    All three carry the identical policy prompt. Stripping it from T3 would
    measure information removal, not where enforcement sits.
    """

    T1_PROMPT = "T1"
    T2_FEW_SHOT = "T2"
    T3_DISPATCH_GUARD = "T3"


class RunManifest(Record):
    """Everything held constant for one run.

    `system_prompt_sha256` identical across T1, T2 and T3 proves no policy
    text was dropped from a treatment. `constraint_set_sha256` proves the guard
    and the scorer read the same rules.
    """

    run_id: NonEmptyStr
    started_at: datetime

    # What was evaluated.
    dataset_version: NonEmptyStr
    dataset_sha256: Sha256
    # Scorer id to version, for every scorer that ran. A version change
    # invalidates dependent baselines.
    scorer_versions: FrozenStrMap = Field(default_factory=FrozenDict)

    # What did the evaluating.
    model: NonEmptyStr
    provider: NonEmptyStr
    treatment: Treatment
    prompt_version: NonEmptyStr
    system_prompt_sha256: Sha256
    constraint_set_sha256: Sha256
    demur_version: NonEmptyStr
    # The decoding parameters the run was configured with; part of baseline
    # comparability. What each call actually sent is `LLMCall.sampling`.
    sampling: Sampling

    # How it was run.
    repeats: int = Field(default=1, ge=1)
    # The harness seed: instance order, fault profiles, stochastic scorers.
    # The model's seed is `sampling.seed`. `None` means not seeded, not seed 0.
    seed: int | None = None
    # Hash of the tolerance bands, absent until they exist. A band edited
    # after a candidate ran shows as a changed hash.
    tolerances_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def check_the_temperature_was_pinned(self) -> Self:
        """Reject a manifest whose sampling has no temperature.

        `None` means the provider chose, and provider defaults move under
        stable aliases, so the run cannot be reproduced. `0` is fine.
        """

        if self.sampling.temperature is None:
            raise ValueError(
                f"run {self.run_id!r} records no temperature, so it cannot be "
                "reproduced. Record the value the adapter sent; 0 is fine."
            )
        return self

    @model_validator(mode="after")
    def check_started_at_is_absolute(self) -> Self:
        """Reject a naive `started_at`.

        Runs are compared across machines and months. A naive timestamp is
        only meaningful on the machine that wrote it.
        """

        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ValueError(
                f"started_at ({self.started_at.isoformat()}) has no timezone. "
                "Record an absolute instant, not a local wall-clock reading."
            )
        return self
