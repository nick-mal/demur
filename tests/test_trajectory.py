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

from demur.sampling import Sampling
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

    Until D-37 commits the first real run, this fixture is the only artifact
    holding the schema still, and regenerating it is a deliberate act rather
    than a forbidden one — nothing published depends on it yet. After D-37 a
    failure here means the schema moved under committed evidence, and the
    answer is a `schema_version` bump with a converting loader, never an edit
    to a file in `runs/`.
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


def local_usage(*, input_tokens: int, output_tokens: int, **kw: int) -> Usage:
    """A normalised report from a provider with no cache rates."""

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=0,
        cache_creation_input_tokens=0,
        **kw,
    )


def test_unreported_usage_is_not_zero() -> None:
    """`None` means the provider said nothing, which is not free."""

    assert Usage().total_tokens is None
    assert local_usage(input_tokens=10, output_tokens=5).total_tokens


@pytest.mark.parametrize(
    "partial",
    [
        {"input_tokens": 10, "output_tokens": 5},
        {"cached_input_tokens": 7},
        {"cache_creation_input_tokens": 7},
        {"reasoning_tokens": 5},
        {"output_tokens": 5, "cached_input_tokens": 0},
        # The bucket a provider without a cache-write premium must still fill.
        {"input_tokens": 10, "output_tokens": 5, "cached_input_tokens": 0},
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

    usage = local_usage(input_tokens=10, output_tokens=5)

    assert usage.reasoning_tokens is None
    assert usage.total_tokens == 15


def test_total_counts_each_prompt_bucket_once_and_excludes_reasoning(
    trajectory: Trajectory,
) -> None:
    usage = trajectory.llm_calls[0].usage

    assert usage.total_tokens == 449


def test_a_local_model_reports_prompt_and_completion_and_nothing_else() -> None:
    """The shape demur's own quick start produces.

    Ollama, llama.cpp and vLLM behind an OpenAI-compatible endpoint report a
    prompt count and a completion count. Both cache buckets are `0` — not
    because no KV cache exists, but because no *rate* does, and these are
    billing buckets. An engine reusing a prefix underneath does not make the
    zero wrong; `economics.py` prices tokens, it does not audit inference
    engines.
    """

    usage = local_usage(input_tokens=1_204, output_tokens=88)

    assert usage.reported
    assert usage.total_tokens == 1_292
    assert usage.cached_input_tokens == 0
    assert usage.cache_creation_input_tokens == 0


def test_cache_writes_are_their_own_bucket() -> None:
    """Folding them into `input_tokens` would keep the total right and lose
    the price: a cache write bills at a premium over ordinary input, so the
    run that populates a cache costs more than the run that reads it. T2 adds
    few-shot exemplars — exactly the content worth caching — which is where
    that difference lands hardest."""

    writing = Usage(
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=0,
        cache_creation_input_tokens=900,
    )
    reading = Usage(
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=900,
        cache_creation_input_tokens=0,
    )

    assert writing.total_tokens == reading.total_tokens == 1_020
    assert writing != reading


def test_reported_collapses_once_validation_has_run() -> None:
    """The breadth of `reported` is for the validator, not for callers.

    It has to see all five fields so `Usage(cached_input_tokens=7)` fails
    rather than passing as "nothing reported". Afterwards only two states
    survive — everything `None`, or every bucket filled — so on a constructed
    `Usage` it says exactly what `input_tokens is not None` says. Pinned so
    that a future field cannot widen the state space unnoticed.
    """

    for usage in (Usage(), local_usage(input_tokens=1, output_tokens=1)):
        assert usage.reported == (usage.input_tokens is not None)


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


def test_a_call_records_the_parameters_it_was_actually_sent_with() -> None:
    """`prompt_at()` plus `model` plus this is the whole request.

    A trajectory has to be readable a year later without the run that produced
    it, so the parameters that shaped a reply cannot live only on the manifest
    — otherwise reproducing one call means finding a second artifact first.
    """

    call = LLMCall(
        index=0,
        model="local:qwen3",
        messages_appended=[Message(role="user", content="total sales in June")],
        response_text="1,963.10",
        sampling=Sampling(temperature=0.7, top_p=0.95, seed=11, stop=("\n\n",)),
    )

    assert call.sampling.temperature == 0.7
    assert LLMCall.model_validate_json(call.model_dump_json()) == call


def test_a_call_may_differ_from_what_the_run_was_configured_with() -> None:
    """The divergence is the evidence, not an inconsistency.

    A retry at a higher temperature after a malformed tool call is a recovery
    strategy demur measures, and a fault profile rewriting a request is one it
    injects. A schema that recorded sampling only once — on the manifest —
    could not express either, and both would read as runs that behaved oddly
    for no recorded reason.
    """

    configured = Sampling(temperature=0)
    retried = Sampling(temperature=0.8)

    assert configured != retried


def test_temperature_zero_is_not_the_same_as_unspecified() -> None:
    """`None` means the provider chose. Zero means we did, and chose greedy.

    Collapsing them would turn "we do not know how this was decoded" into "this
    was deterministic" — the same failure `Usage` guards against when it
    refuses to read an unreported token count as free.
    """

    assert Sampling(temperature=0).specified is True
    assert Sampling().specified is False
    assert Sampling(temperature=0) != Sampling()


def test_reasoning_survives_into_a_rebuilt_prompt() -> None:
    """A provider continuing a turn that contained thinking expects it back.

    This is why `reasoning_text` is on `Message` as well as on `LLMCall`: the
    reply is rendered into a turn by `assistant_turn`, and a rebuilt prompt
    that dropped the thinking would not be the conversation that happened.
    """

    call = LLMCall(
        index=0,
        model="local:qwen3",
        messages_appended=[Message(role="user", content="total sales in June")],
        reasoning_text="Two date columns could anchor 'June'. Neither is named.",
        response_text="Which date should I anchor on?",
    )

    turn = call.assistant_turn

    assert turn is not None
    assert turn.reasoning_text == call.reasoning_text
    assert turn.content == "Which date should I anchor on?"


def test_thinking_alone_is_not_a_turn() -> None:
    """Reasoning is what led to a turn, not a turn.

    A model that only thought and neither spoke nor called a tool has produced
    nothing to append to the conversation, so it contributes no message — the
    same rule that keeps a provider error from contributing one.
    """

    call = LLMCall(
        index=0,
        model="local:qwen3",
        messages_appended=[Message(role="user", content="hello")],
        reasoning_text="thinking about it",
    )

    assert call.assistant_turn is None

    with pytest.raises(ValidationError, match="neither content nor tool calls"):
        Message(role="assistant", reasoning_text="thinking about it")


def test_only_the_model_reasons() -> None:
    """Attributing thinking to a user or a tool result would put words in the
    wrong mouth on replay."""

    for role in ("user", "system", "tool"):
        with pytest.raises(ValidationError, match="carries reasoning text"):
            Message(
                role=role,
                content="text",
                reasoning_text="not mine",
                tool_call_id="call-1" if role == "tool" else None,
            )


def test_reasoning_tokens_without_text_is_the_normal_case() -> None:
    """No validator ties `reasoning_text` to `usage.reasoning_tokens`.

    They vary independently in both directions, and a rule demanding both
    would reject two correct adapters. Current Anthropic models bill thinking
    while returning no raw chain of thought at all — tokens without text is
    what a correct adapter records, not a recording bug. A local model behind
    vLLM does the reverse: it returns `reasoning_content` while reporting no
    separate token count.
    """

    billed_but_hidden = LLMCall(
        index=0,
        model="claude-opus-5",
        messages_appended=[Message(role="user", content="q")],
        response_text="a",
        usage=Usage(
            input_tokens=10,
            output_tokens=50,
            cached_input_tokens=0,
            cache_creation_input_tokens=0,
            reasoning_tokens=40,
        ),
    )
    text_but_uncounted = LLMCall(
        index=0,
        model="local:qwen3",
        messages_appended=[Message(role="user", content="q")],
        response_text="a",
        reasoning_text="<think>weighing the two readings</think>",
        usage=local_usage(input_tokens=10, output_tokens=50),
    )

    assert billed_but_hidden.reasoning_text is None
    assert billed_but_hidden.usage.reasoning_tokens == 40
    assert text_but_uncounted.reasoning_text is not None
    assert text_but_uncounted.usage.reasoning_tokens is None


def test_an_additive_field_does_not_orphan_the_golden_fixture() -> None:
    """The rule `SCHEMA_VERSION` states, exercised on a live example.

    `reasoning_text` is optional with a default, so the committed fixture —
    written before it existed and carrying no such key — still validates and
    still compares equal. That is what makes it additive, and why it must not
    spend a schema version: a bump has to mean "this build cannot read that
    file" or the signal is worthless.
    """

    payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
    calls = [step for step in payload["steps"] if step["kind"] == "llm_call"]

    assert calls
    assert all("reasoning_text" not in call for call in calls)
    assert all(
        call.reasoning_text is None for call in load_trajectory(payload).llm_calls
    )
