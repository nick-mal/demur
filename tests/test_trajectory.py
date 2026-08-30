"""The trajectory is only evidence if it survives a round trip — and a release.

Everything published from this project is recomputed from committed
trajectories, so serialise → deserialise → compare equal is the load-bearing
test here, and the golden fixture is the one that catches a schema change made
without bumping `schema_version`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from demur.trajectory import (
    SCHEMA_VERSION,
    LLMCall,
    Message,
    ToolCall,
    ToolOutcome,
    ToolRequest,
    Trajectory,
    UnknownSchemaVersionError,
    Usage,
    load_trajectory,
)

GOLDEN = Path(__file__).parent / "fixtures" / "trajectory-v1.json"


def test_round_trips_through_json(trajectory: Trajectory) -> None:
    original = trajectory

    assert load_trajectory(original.model_dump_json()) == original


def test_the_golden_fixture_still_loads_and_matches(trajectory: Trajectory) -> None:
    """A committed run must outlive the code that wrote it.

    If this fails, either the schema changed without a `schema_version` bump
    and a loader for version 1, or the fixture was edited to suit new code.
    Neither is allowed: the fixture stands in for every run in `runs/`.
    """

    assert load_trajectory(GOLDEN.read_text(encoding="utf-8")) == (trajectory)


def test_the_golden_fixture_is_written_at_the_current_version() -> None:
    assert json.loads(GOLDEN.read_text(encoding="utf-8"))["schema_version"] == (
        SCHEMA_VERSION
    )


def test_load_accepts_text_bytes_and_mappings(trajectory: Trajectory) -> None:
    payload = trajectory.model_dump(mode="json")
    text = json.dumps(payload)

    assert load_trajectory(text) == load_trajectory(text.encode("utf-8"))
    assert load_trajectory(payload) == load_trajectory(text)


def test_an_unknown_schema_version_says_so(trajectory: Trajectory) -> None:
    """Not a pile of field errors that read like corrupt data."""

    payload = trajectory.model_dump(mode="json")
    payload["schema_version"] = "99"

    with pytest.raises(UnknownSchemaVersionError, match="cannot read trajectory"):
        load_trajectory(payload)


def test_a_missing_schema_version_is_reported_as_missing(
    trajectory: Trajectory,
) -> None:
    payload = trajectory.model_dump(mode="json")
    del payload["schema_version"]

    with pytest.raises(UnknownSchemaVersionError, match="no schema_version"):
        load_trajectory(payload)


def test_steps_keep_their_type_across_the_round_trip(trajectory: Trajectory) -> None:
    """`kind` is what tells a tool call from an LLM call on the way back."""

    restored = load_trajectory(trajectory.model_dump_json())

    assert [type(step) for step in restored.steps] == [
        LLMCall,
        ToolCall,
        LLMCall,
        ToolCall,
        LLMCall,
    ]
    assert restored.tool_calls[0].arguments["dry_run"] is False


def test_step_indices_must_match_their_position(trajectory: Trajectory) -> None:
    payload = trajectory.model_dump(mode="json")
    payload["steps"][2]["index"] = 7

    with pytest.raises(ValidationError, match="carries index 7"):
        load_trajectory(payload)


def test_blocked_calls_are_visible(trajectory: Trajectory) -> None:
    """The guard is measured by what it prevented, so refusals leave a trace."""

    blocked = trajectory.blocked_calls

    assert [call.name for call in blocked] == ["run_query"]
    assert blocked[0].blocked_by == "describe_before_query"
    assert blocked[0].outcome.result is None


def test_a_blocked_call_must_name_the_constraint_that_blocked_it() -> None:
    with pytest.raises(ValidationError, match="names no constraint"):
        ToolCall(
            index=0,
            name="run_query",
            call_id="call-1",
            outcome=ToolOutcome(status="blocked"),
        )


def test_a_call_that_ran_cannot_name_a_blocking_constraint() -> None:
    with pytest.raises(ValidationError, match="must not have executed"):
        ToolCall(
            index=0,
            name="run_query",
            call_id="call-1",
            outcome=ToolOutcome(status="ok", result=1),
            blocked_by="describe_before_query",
        )


def test_error_outcomes_carry_a_message_and_others_do_not() -> None:
    with pytest.raises(ValidationError, match="error message"):
        ToolOutcome(status="error")

    with pytest.raises(ValidationError, match="must not carry an error message"):
        ToolOutcome(status="ok", error="boom")


def test_a_blocked_outcome_has_no_result() -> None:
    with pytest.raises(ValidationError, match="the tool never ran"):
        ToolOutcome(status="blocked", result={"rows": []})


def test_unreported_usage_is_not_zero() -> None:
    """`None` means the provider said nothing, which is not free."""

    assert Usage().total_tokens is None
    assert Usage(input_tokens=10, output_tokens=5, cached_input_tokens=0).total_tokens


@pytest.mark.parametrize(
    "partial",
    [
        {"input_tokens": 10, "output_tokens": 5},
        {"cached_input_tokens": 7},
        {"reasoning_tokens": 5},
        {"output_tokens": 5, "cached_input_tokens": 0},
    ],
)
def test_half_reported_usage_is_rejected(partial: dict[str, int]) -> None:
    """The accident that would blind cost accounting without a peep.

    An adapter that reports some counts but forgets a bucket would make
    `total_tokens` `None` on every call while every other field looked healthy.
    Any field being set means the provider reported — including the two that
    are easy to overlook.
    """

    with pytest.raises(ValidationError, match="unreported"):
        Usage(**partial)


def test_a_provider_that_says_nothing_is_still_legal() -> None:
    assert Usage().total_tokens is None
    assert not Usage().reported


def test_reasoning_stays_optional_within_a_reported_usage() -> None:
    """Most providers report no reasoning bucket at all."""

    usage = Usage(input_tokens=10, output_tokens=5, cached_input_tokens=0)

    assert usage.reasoning_tokens is None
    assert usage.total_tokens == 15


def test_total_counts_each_prompt_bucket_once_and_excludes_reasoning(
    trajectory: Trajectory,
) -> None:
    usage = trajectory.llm_calls[0].usage

    assert usage.total_tokens == 449


def test_reasoning_cannot_exceed_output() -> None:
    with pytest.raises(ValidationError, match="not a bucket beside it"):
        Usage(output_tokens=10, reasoning_tokens=11)


def test_token_counts_cannot_be_negative() -> None:
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1)


def test_the_prompt_is_the_request_and_stops_there(trajectory: Trajectory) -> None:
    """Deltas are acceptable only because the request rebuilds exactly.

    The boundary is the point: a prompt that included the call's own reply
    would ask a provider to continue past its own answer, which is not the
    request that was made.
    """

    assert [message.role for message in trajectory.prompt_at(0)] == ["system", "user"]

    # The last call asked the model to answer, given the tool result it had.
    assert [message.role for message in trajectory.prompt_at(4)] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]


def test_the_prompt_carries_the_earlier_replies_verbatim(
    trajectory: Trajectory,
) -> None:
    """Assistant turns are derived from the reply, not stored a second time."""

    prompt = trajectory.prompt_at(4)
    first_reply = trajectory.llm_calls[0].assistant_turn

    assert prompt[2] == first_reply
    assert first_reply is not None
    assert first_reply.tool_calls[0].call_id == "call-1"
    assert prompt[-1].role == "tool"


def test_a_call_that_produced_nothing_contributes_no_turn() -> None:
    """A provider error adds nothing to the conversation."""

    call = LLMCall(
        index=0,
        model="local:qwen3",
        messages_appended=[Message(role="user", content="hi")],
    )

    assert call.assistant_turn is None


def test_a_reply_stored_in_its_own_delta_is_rejected() -> None:
    """The duplication that would make prompt_at overshoot by a turn."""

    with pytest.raises(ValidationError, match="a call's own reply belongs in"):
        LLMCall(
            index=0,
            model="local:qwen3",
            messages_appended=[
                Message(role="user", content="total sales in June"),
                Message(
                    role="assistant",
                    tool_calls=[ToolRequest(call_id="call-1", name="run_query")],
                ),
            ],
            tool_calls_requested=[ToolRequest(call_id="call-1", name="run_query")],
        )


def test_a_negative_index_is_an_error_not_an_empty_prompt(
    trajectory: Trajectory,
) -> None:
    """`steps[-1]` is an LLM call here, so this would silently return `()`."""

    with pytest.raises(IndexError):
        trajectory.prompt_at(-1)


def test_asking_a_tool_step_for_a_prompt_is_an_error(trajectory: Trajectory) -> None:
    with pytest.raises(TypeError, match="not an LLM call"):
        trajectory.prompt_at(1)


def test_requests_pair_with_dispatches_on_call_id(trajectory: Trajectory) -> None:
    """Invalid-attempt rate is model intent, measured before the guard acts."""

    requested = [
        request
        for call in trajectory.llm_calls
        for request in call.tool_calls_requested
    ]

    assert [(r.call_id, r.name) for r in requested] == [
        ("call-1", "run_query"),
        ("call-2", "describe_schema"),
    ]
    dispatched = {call.call_id: call for call in trajectory.tool_calls}
    assert dispatched["call-1"].blocked
    assert not dispatched["call-2"].blocked


def test_two_dispatches_cannot_share_a_call_id(trajectory: Trajectory) -> None:
    """A duplicated id must read as two calls, not one collapsed into another."""

    payload = trajectory.model_dump(mode="json")
    payload["steps"][3]["call_id"] = "call-1"

    with pytest.raises(ValidationError, match="more than one tool call"):
        load_trajectory(payload)


def test_unparseable_arguments_are_recorded_rather_than_flattened() -> None:
    """Fault profile 3 injects exactly this; an empty dict would lose it."""

    call = ToolCall(
        index=0,
        name="run_query",
        call_id="call-1",
        raw_arguments='{"sql": "SELECT',
        outcome=ToolOutcome(status="error", error="invalid JSON in arguments"),
    )

    assert call.arguments == {}
    assert call.raw_arguments == '{"sql": "SELECT'


def test_a_call_cannot_be_both_parsed_and_unparsed() -> None:
    with pytest.raises(ValidationError, match="one of them is a fiction"):
        ToolCall(
            index=0,
            name="run_query",
            call_id="call-1",
            arguments={"sql": "SELECT 1"},
            raw_arguments='{"sql": "SELECT',
            outcome=ToolOutcome(status="error", error="invalid JSON"),
        )


def test_a_call_the_dispatcher_could_not_read_cannot_have_succeeded() -> None:
    with pytest.raises(ValidationError, match="could not be parsed"):
        ToolCall(
            index=0,
            name="run_query",
            call_id="call-1",
            raw_arguments='{"sql": "SELECT',
            outcome=ToolOutcome(status="ok", result={"rows": []}),
        )


def test_an_assistant_turn_can_carry_tool_calls_without_content() -> None:
    """What every OpenAI-compatible provider emits for a tool-calling turn."""

    turn = Message(
        role="assistant",
        tool_calls=[ToolRequest(call_id="call-1", name="run_query")],
    )

    assert turn.content is None
    assert turn.tool_calls[0].name == "run_query"


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_an_empty_turn_is_rejected(blank: str | None) -> None:
    """Blank content is no content — an empty system prompt is a bug, not a turn."""

    with pytest.raises(ValidationError, match="recording bug"):
        Message(role="system", content=blank)


def test_a_requested_call_must_be_pairable() -> None:
    """Both ends of the pairing are policed, not just the tool-result end."""

    with pytest.raises(ValidationError):
        ToolRequest(name="run_query")

    with pytest.raises(ValidationError):
        ToolRequest(call_id="", name="run_query")


def test_only_assistants_request_tools_and_tool_results_name_their_call() -> None:
    with pytest.raises(ValidationError, match="cannot request tool calls"):
        Message(
            role="user",
            content="hi",
            tool_calls=[ToolRequest(call_id="call-1", name="x")],
        )

    with pytest.raises(ValidationError, match="must name the call it answers"):
        Message(role="tool", content="{}")


def test_a_completed_run_must_have_answered(trajectory: Trajectory) -> None:
    """`completed` means the agent answered; anything else is another state."""

    payload = trajectory.model_dump(mode="json")
    payload["final_answer"] = None

    with pytest.raises(ValidationError, match="no final answer"):
        load_trajectory(payload)


def test_a_run_that_did_nothing_cannot_have_completed_or_escalated() -> None:
    with pytest.raises(ValidationError, match="the agent did nothing"):
        Trajectory(
            run_id="run-1",
            instance_id="rs-014",
            terminal_state="completed",
            final_answer="anything",
        )

    with pytest.raises(ValidationError, match="the agent did nothing"):
        Trajectory(run_id="run-1", instance_id="rs-014", terminal_state="escalated")


def test_only_a_completed_run_carries_a_final_answer(trajectory: Trajectory) -> None:
    """A run that escalated or ran out of steps stopped without answering."""

    payload = trajectory.model_dump(mode="json")
    payload["terminal_state"] = "step_budget_exhausted"

    with pytest.raises(ValidationError, match="carries a final answer"):
        load_trajectory(payload)


def test_escalation_termination_is_checked_against_a_named_tool(
    trajectory: Trajectory,
) -> None:
    """Which tool escalates is domain knowledge, so the library is told it.

    Hard-coding `escalate` here would put a tool name from the example inside
    the domain-blind library; the `Terminal` constraint supplies it instead.
    """

    assert not trajectory.ends_with_tool("escalate")
    assert trajectory.ends_with_tool("describe_schema") is False  # not the last step


def test_identifiers_must_identify_something() -> None:
    """An empty run_id becomes a `runs/` directory with no name."""

    with pytest.raises(ValidationError):
        Trajectory(run_id="", instance_id="rs-014", terminal_state="provider_error")

    with pytest.raises(ValidationError):
        Trajectory(run_id="run-1", instance_id="  ", terminal_state="provider_error")


def test_prefix_is_what_a_constraint_sees(trajectory: Trajectory) -> None:

    assert trajectory.prefix(0) == ()
    assert len(trajectory.prefix(2)) == 2
    assert trajectory.prefix(2)[-1].name == "run_query"
    assert len(trajectory.prefix(99)) == len(trajectory.steps)

    with pytest.raises(IndexError):
        trajectory.prefix(-1)


def test_trajectories_are_immutable(trajectory: Trajectory) -> None:
    """Scoring reads the record; it does not get to annotate it."""

    with pytest.raises(ValidationError):
        trajectory.instance_id = "something-else"


def test_unknown_fields_are_rejected(trajectory: Trajectory) -> None:
    """A drifting field name fails at load instead of vanishing quietly."""

    payload = trajectory.model_dump(mode="json")
    payload["wall_time_ms"] = 10.0

    with pytest.raises(ValidationError):
        load_trajectory(payload)
