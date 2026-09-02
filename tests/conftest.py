"""Shared test data.

One hand-written trajectory, used by every module that needs a realistic
record: a blocked call and the recovery after it are the shapes the enforcement
finding is measured from, and the committed golden fixture mirrors this exactly.
"""

from __future__ import annotations

import pytest

from demur.trajectory import (
    LLMCall,
    Message,
    OutcomeStatus,
    TerminalState,
    ToolCall,
    ToolOutcome,
    ToolRequest,
    Trajectory,
    Usage,
)

ANSWER = "June sales totalled 1,963.10."


def hand_written_trajectory() -> Trajectory:
    """A run that is blocked, recovers, and then answers."""

    run_query = ToolRequest(
        call_id="call-1",
        name="run_query",
        arguments={"sql": "SELECT sum(total) FROM orders", "dry_run": False},
    )
    describe = ToolRequest(
        call_id="call-2", name="describe_schema", arguments={"table": "orders"}
    )

    return Trajectory(
        run_id="run-2026-08-29-0001",
        instance_id="rs-014",
        terminal_state=TerminalState.COMPLETED,
        final_answer=ANSWER,
        wall_ms=2342.5,
        provider_meta={"provider": "local", "alias": "qwen3"},
        steps=[
            LLMCall(
                index=0,
                model="local:qwen3",
                messages_appended=[
                    Message(role="system", content="policy text"),
                    Message(role="user", content="total sales in June"),
                ],
                tool_calls_requested=[run_query],
                finish_reason="tool_calls",
                usage=Usage(
                    input_tokens=412,
                    output_tokens=37,
                    cached_input_tokens=0,
                    cache_creation_input_tokens=0,
                    reasoning_tokens=12,
                ),
                latency_ms=903.0,
                provider_request_id="req-a1",
            ),
            ToolCall(
                index=1,
                name="run_query",
                arguments=run_query.arguments,
                outcome=ToolOutcome(status=OutcomeStatus.BLOCKED),
                blocked_by="describe_before_query",
                call_id="call-1",
            ),
            LLMCall(
                index=2,
                model="local:qwen3",
                messages_appended=[
                    Message(
                        role="tool",
                        content="blocked by describe_before_query",
                        tool_call_id="call-1",
                    ),
                ],
                tool_calls_requested=[describe],
                finish_reason="tool_calls",
                usage=Usage(
                    input_tokens=56,
                    output_tokens=21,
                    cached_input_tokens=412,
                    cache_creation_input_tokens=0,
                ),
                latency_ms=611.0,
                provider_request_id="req-a2",
            ),
            ToolCall(
                index=3,
                name="describe_schema",
                arguments={"table": "orders"},
                outcome=ToolOutcome(
                    status=OutcomeStatus.OK,
                    result={"columns": ["id", "total"], "restricted": False},
                ),
                call_id="call-2",
                latency_ms=4.25,
            ),
            LLMCall(
                index=4,
                model="local:qwen3",
                messages_appended=[
                    Message(
                        role="tool",
                        content='{"columns": ["id", "total"], "restricted": false}',
                        tool_call_id="call-2",
                    ),
                ],
                response_text=ANSWER,
                finish_reason="stop",
                usage=Usage(
                    input_tokens=64,
                    output_tokens=18,
                    cached_input_tokens=468,
                    cache_creation_input_tokens=0,
                ),
                latency_ms=502.0,
                provider_request_id="req-a3",
            ),
        ],
    )


@pytest.fixture
def trajectory() -> Trajectory:
    return hand_written_trajectory()
