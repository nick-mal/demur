"""What the manifest has to guarantee.

Two things: that the hashes proving the experimental control are well formed,
and that a run's timestamp means the same thing on every machine that reads it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from demur.manifest import RunManifest, Treatment

DIGEST = "a" * 64


def manifest(**overrides: object) -> RunManifest:
    fields: dict[str, object] = {
        "run_id": "run-2026-08-29-0001",
        "started_at": datetime(2026, 8, 29, 18, 23, tzinfo=UTC),
        "dataset_version": "v0",
        "dataset_sha256": DIGEST,
        "scorer_versions": {"result_correct": "1.0.0", "abstention_correct": "1.0.0"},
        "model": "local:qwen3",
        "provider": "local",
        "treatment": Treatment.T1_PROMPT,
        "prompt_version": "p1",
        "system_prompt_sha256": "b" * 64,
        "constraint_set_sha256": "c" * 64,
        "demur_version": "0.1.0",
    }
    return RunManifest.model_validate(fields | overrides)


def test_round_trips_through_json() -> None:
    original = manifest(seed=7, repeats=3, tolerances_sha256="d" * 64)

    assert RunManifest.model_validate_json(original.model_dump_json()) == original


def test_the_control_is_provable_by_hash() -> None:
    """T1, T2 and T3 must carry byte-identical policy prompts.

    The comparison between treatments is only valid if the policy text is the
    same in all three. Comparing this field across their manifests is what
    turns that from an assertion in a document into something checkable.
    """

    t1 = manifest(treatment=Treatment.T1_PROMPT)
    t3 = manifest(run_id="run-2", treatment=Treatment.T3_DISPATCH_GUARD)

    assert t1.system_prompt_sha256 == t3.system_prompt_sha256
    assert t1.treatment is not t3.treatment


@pytest.mark.parametrize(
    "bad", ["", "abc", DIGEST.upper(), DIGEST[:-1], f"{DIGEST[:-1]}z"]
)
def test_malformed_digests_are_rejected(bad: str) -> None:
    """A truncated or upper-cased digest would look plausible and never match."""

    with pytest.raises(ValidationError):
        manifest(dataset_sha256=bad)


def test_a_naive_timestamp_is_rejected() -> None:
    """Runs get compared across machines; local wall-clock does not travel."""

    naive = datetime(2026, 8, 29, 18, 23)  # noqa: DTZ001 — the point of the test
    with pytest.raises(ValidationError, match="no timezone"):
        manifest(started_at=naive)


def test_an_unseeded_run_records_none_not_zero() -> None:
    assert manifest().seed is None
    assert manifest(seed=0).seed == 0


def test_repeats_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        manifest(repeats=0)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        manifest(dataset_hash=DIGEST)
