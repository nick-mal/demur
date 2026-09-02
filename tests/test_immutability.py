"""A record must be immutable all the way down, not just at the top.

`frozen=True` stops `traj.instance_id = ...` and nothing else. The mutations
that happen by accident go through a field: a scorer stashing a value in
`provider_meta`, a helper appending a step, a retry rewriting `arguments`.
These tests are the executable half of specification §4's first invariant.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from demur.manifest import RunManifest, Treatment
from demur.record import FrozenDict, FrozenList
from demur.runner.dataset import (
    Expected,
    Instance,
    Interpretation,
    Origin,
    Resolution,
    Split,
)
from demur.sampling import Sampling
from demur.trajectory import Trajectory


def instance() -> Instance:
    return Instance(
        id="rs-042",
        request="What were sales in June?",
        fixture_state={"schema": "regional_sales", "seeded": [1, {"rows": 2}]},
        constraints=("describe_before_query",),
        expected=Expected(resolution=Resolution.ESCALATE, reason="four date columns"),
        interpretations=(
            Interpretation(
                id="order_date",
                description="on OrderDate",
                reference_output={"rows": [[1]]},
            ),
        ),
        split=Split.TEST,
        origin=Origin.DRAFTED,
    )


def manifest() -> RunManifest:
    return RunManifest(
        run_id="run-1",
        started_at=datetime(2026, 8, 30, 9, 0, tzinfo=UTC),
        dataset_version="v0",
        dataset_sha256="a" * 64,
        scorer_versions={"result_correct": "1.0.0"},
        model="local:qwen3",
        provider="local",
        treatment=Treatment.T1_PROMPT,
        prompt_version="p1",
        system_prompt_sha256="b" * 64,
        constraint_set_sha256="c" * 64,
        sampling=Sampling(temperature=0),
        demur_version="0.1.1",
    )


def test_the_step_list_cannot_be_grown_or_shortened(trajectory: Trajectory) -> None:
    """Declared sequences are tuples, so the list API is simply absent."""

    steps = trajectory.steps

    assert isinstance(steps, tuple)
    with pytest.raises(AttributeError, match="'tuple' object has no attribute"):
        steps.append(steps[0])  # pyright: ignore[reportAttributeAccessIssue]


def test_provider_meta_cannot_be_annotated(trajectory: Trajectory) -> None:
    """The mutation a scorer would make by accident."""

    with pytest.raises(TypeError, match="never annotate"):
        trajectory.provider_meta["annotated_by"] = "scorer"


def test_tool_arguments_cannot_be_rewritten(trajectory: Trajectory) -> None:
    """What the agent asked for is the measurement; editing it forges data."""

    call = trajectory.tool_calls[0]

    with pytest.raises(TypeError, match="immutable"):
        call.arguments["sql"] = "SELECT 1"


def test_the_prompt_cannot_be_edited(trajectory: Trajectory) -> None:
    """The prompt is the experimental manipulation, so it is part of the record."""

    messages = trajectory.llm_calls[0].messages_appended

    assert isinstance(messages, tuple)
    with pytest.raises(AttributeError, match="'tuple' object has no attribute"):
        messages.clear()  # pyright: ignore[reportAttributeAccessIssue]


def test_nested_json_is_frozen_at_every_level(trajectory: Trajectory) -> None:
    """A top-level-only guard still lets a nested container be edited."""

    nested = Trajectory.model_validate(
        trajectory.model_dump(mode="json")
        | {"provider_meta": {"retries": [{"attempt": 1, "tags": ["a"]}]}}
    )
    retries = nested.provider_meta["retries"]

    assert isinstance(retries, FrozenList)
    assert isinstance(retries[0], FrozenDict)
    with pytest.raises(TypeError):
        retries[0]["tags"].append("b")


def test_a_defaulted_mapping_is_frozen_too(trajectory: Trajectory) -> None:
    """Defaults skip validation, so an unfrozen default would be a live hole."""

    payload = trajectory.model_dump(mode="json")
    del payload["provider_meta"]

    with pytest.raises(TypeError):
        Trajectory.model_validate(payload).provider_meta["x"] = 1


def test_instances_are_frozen_all_the_way_down() -> None:
    case = instance()

    with pytest.raises(AttributeError):
        case.interpretations.append(  # pyright: ignore[reportAttributeAccessIssue]
            Interpretation(id="ship_date", description="on ShipDate")
        )
    seeded = case.fixture_state["seeded"]
    assert isinstance(seeded, list)
    assert isinstance(seeded[1], dict)
    with pytest.raises(TypeError):
        seeded[1]["rows"] = 3

    output = case.interpretations[0].reference_output
    assert isinstance(output, dict)
    with pytest.raises(TypeError):
        output["rows"] = []


def test_manifest_scorer_versions_cannot_be_edited() -> None:
    """A version change invalidates baselines; it must not be editable in place."""

    with pytest.raises(TypeError):
        manifest().scorer_versions["result_correct"] = "9.9.9"


def test_frozen_containers_still_compare_equal_to_plain_ones(
    trajectory: Trajectory,
) -> None:
    """Freezing must not change what a record is, only what can be done to it.
    Round-trip equality and content hashing both rest on this."""

    assert trajectory.provider_meta == {"provider": "local", "alias": "qwen3"}
    assert trajectory.tool_calls[1].arguments == {"table": "orders"}
    assert trajectory.llm_calls[0].tool_calls_requested[0].arguments == {
        "sql": "SELECT sum(total) FROM orders",
        "dry_run": False,
    }
