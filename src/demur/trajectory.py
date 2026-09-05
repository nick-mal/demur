"""The canonical record of what an agent did. See spec §4, Trajectory.

Everything downstream reads it, and published results are recomputed from
committed copies. So every model here round-trips losslessly through JSON,
and `schema_version` keeps old files readable after the code moves on.

Prompts are stored as deltas. An `LLMCall` records what arrived before it,
never its own reply, and `prompt_at()` rebuilds a request by interleaving
deltas with the replies between them. Exact only while the loop appends: a
loop that rewrites history needs a new schema version.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from demur.record import (
    FrozenDict,
    FrozenJson,
    FrozenJsonObject,
    NonEmptyStr,
    Record,
    TokenCount,
)
from demur.sampling import Sampling

SCHEMA_VERSION = "1"
"""The schema version written into every new trajectory.

Bump when an older committed trajectory would no longer load. An optional
field with a default is additive and does not bump; a field joining a required
set does. Until a run is committed there is no evidence to protect, so a
breaking change rewrites the schema in place.
"""

_USAGE_BUCKETS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
)
"""Every bucket a reported usage must fill. Reasoning is inside output."""


class Usage(Record):
    """Token counts for one call, in billing buckets. See spec §4, Usage.

    Each field exists because some provider prices those tokens differently.
    The three prompt buckets are disjoint; `reasoning_tokens` is inside
    `output_tokens`. `None` means not reported, which is not zero. No cost
    field: prices are applied at reporting time from a versioned table.
    """

    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    # Prompt tokens served from cache, billed at a discount where one exists.
    cached_input_tokens: TokenCount | None = None
    # Prompt tokens written into a cache, billed at a premium where one exists.
    cache_creation_input_tokens: TokenCount | None = None
    reasoning_tokens: TokenCount | None = None

    @property
    def reported(self) -> bool:
        """Return whether the provider reported any count at all.

        Any field counts, so a half-report is visible to the validator.
        """

        return any(getattr(self, name) is not None for name in type(self).model_fields)

    @property
    def total_tokens(self) -> int | None:
        """Return prompt buckets plus output, or `None` if nothing was reported."""

        if not self.reported:
            return None
        # The validator below guarantees every bucket is set.
        return sum(getattr(self, name) for name in _USAGE_BUCKETS)

    @model_validator(mode="after")
    def check_reasoning_fits_inside_output(self) -> Self:
        """Reject reasoning tokens that exceed output tokens.

        Reasoning is the thinking part of the output, not a bucket beside it.
        """

        if self.reasoning_tokens is None or self.output_tokens is None:
            return self
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError(
                f"reasoning_tokens ({self.reasoning_tokens}) exceeds output_tokens "
                f"({self.output_tokens}), but reasoning is part of output, not a "
                "bucket beside it. The adapter must add reasoning into output."
            )
        return self

    @model_validator(mode="after")
    def check_usage_is_all_or_nothing(self) -> Self:
        """Reject a usage that reports some buckets and not others.

        A missing bucket would silently drop tokens from `total_tokens`.
        """

        if not self.reported:
            return self
        missing = [name for name in _USAGE_BUCKETS if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"usage reports some counts but leaves {', '.join(missing)} "
                "unreported. Write 0 where the provider has no such bucket; "
                "`None` means it said nothing at all."
            )
        return self


class _ToolInvocation(Record):
    """Fields shared by a tool request and its dispatch."""

    # Pairs a request with its dispatch. Required: an unpaired call is not
    # evidence. A provider that emits no id leaves the adapter to make one.
    call_id: NonEmptyStr
    name: NonEmptyStr
    arguments: FrozenJsonObject = Field(default_factory=FrozenDict)
    # The argument text as emitted, recorded only when it could not be parsed.
    # Fault profile 3 injects this; flattening it to `{}` would lose the evidence.
    raw_arguments: str | None = None

    @model_validator(mode="after")
    def check_unparsed_arguments_are_not_also_parsed(self) -> Self:
        """Reject parsed arguments beside `raw_arguments`.

        Both set means one of them is invented.
        """

        if self.raw_arguments is not None and self.arguments:
            raise ValueError(
                f"{self.name!r} carries both parsed arguments and raw_arguments, so "
                "one of them is a fiction. raw_arguments records only text that "
                "could not be parsed."
            )
        return self


class ToolRequest(_ToolInvocation):
    """A tool call as the model asked for it, before dispatch.

    Separate from `ToolCall` because the invalid-attempt rate is model intent,
    unaffected by whether the guard let the call through.
    """


class Message(Record):
    """One entry of the conversation as sent to the provider.

    Faithful enough to replay: an assistant turn that only calls tools has no
    content and carries the calls, which is what every provider emits.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    # Assistant turns only: what the model thought. Carried so a rebuilt prompt
    # can be replayed; a provider continuing such a turn expects it back.
    reasoning_text: str | None = None
    # Assistant turns only: the calls this turn asked for.
    tool_calls: tuple[ToolRequest, ...] = ()
    # Tool turns only: which `ToolRequest.call_id` this result answers.
    tool_call_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def check_shape_matches_role(self) -> Self:
        """Reject fields that do not belong to the message's role.

        Thinking alone is not a turn: a model that only thought contributes
        no message.
        """

        if self.role != "assistant" and self.tool_calls:
            raise ValueError(f"a {self.role} message cannot request tool calls")
        if self.role != "assistant" and self.reasoning_text is not None:
            raise ValueError(
                f"a {self.role} message carries reasoning text, but only the model "
                "reasons. Attributing it elsewhere puts words in the wrong mouth."
            )
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError(
                "a tool message must name the call it answers. An unattributed "
                "result cannot be paired with its request."
            )
        if not (self.content or "").strip() and not self.tool_calls:
            raise ValueError(
                f"a {self.role} message has neither content nor tool calls, which "
                "is a recording bug. Blank content counts as none; a tool with "
                "nothing to say returns a representation of nothing."
            )
        return self


class Completion(Record):
    """What a provider returned for one request. See spec §5, Provider.

    Everything about the reply and nothing about its place in the run. The
    agent loop adds `index` and the delta to make an `LLMCall`.
    """

    model: NonEmptyStr
    response_text: str | None = None
    # What the model thought, beside the content as every provider reports it.
    # `None` does not mean it did not think; unrelated to `reasoning_tokens`.
    reasoning_text: str | None = None
    # Each request that reached the dispatcher also appears as a `ToolCall`,
    # paired on `call_id`, including one the guard blocked.
    tool_calls_requested: tuple[ToolRequest, ...] = ()
    finish_reason: str | None = None
    # What was actually sent. `RunManifest.sampling` is what was configured.
    sampling: Sampling = Field(default_factory=Sampling)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float | None = Field(default=None, ge=0)
    provider_request_id: str | None = None

    @property
    def answered(self) -> bool:
        """Return whether the model replied in prose and asked for no tool.

        That is the agent giving its answer, and the loop stops there.
        """

        return not self.tool_calls_requested and bool(
            (self.response_text or "").strip()
        )

    @property
    def assistant_turn(self) -> Message | None:
        """Return the reply as an assistant `Message`, or `None` if it was empty.

        Derived, not stored, so a replay and the invalid-attempt count read
        the same thing. Not byte-exact against Anthropic: thinking-block
        signatures are not modelled.
        """

        if not (self.response_text or "").strip() and not self.tool_calls_requested:
            return None
        return Message(
            role="assistant",
            content=self.response_text,
            reasoning_text=self.reasoning_text,
            tool_calls=self.tool_calls_requested,
        )


class LLMCall(Completion):
    """One completion request and what came back.

    `messages_appended` is the delta that arrived before this call, never its
    own reply. Prompts are recorded rather than regenerated because the policy
    prompt is the experimental manipulation.
    """

    kind: Literal["llm_call"] = "llm_call"
    index: int = Field(ge=0)
    messages_appended: tuple[Message, ...]

    @model_validator(mode="after")
    def check_the_reply_is_not_also_in_the_delta(self) -> Self:
        """Reject a delta that ends with an assistant turn.

        That is where this call's own reply would land, stored twice. Few-shot
        exemplars are fine; they are never last.
        """

        if self.messages_appended and self.messages_appended[-1].role == "assistant":
            raise ValueError(
                "the messages before an LLM call end with an assistant turn, but "
                "a call's own reply belongs in response_text and "
                "tool_calls_requested. Do not store it in the delta."
            )
        return self


class OutcomeStatus(StrEnum):
    """How a tool call ended."""

    OK = "ok"
    ERROR = "error"
    # Refused by the guard before execution. Not an error: the tool never ran.
    BLOCKED = "blocked"


class ToolOutcome(Record):
    """What came of a tool call, including one that never ran."""

    status: OutcomeStatus
    result: FrozenJson = None
    error: str | None = None

    @model_validator(mode="after")
    def check_fields_match_status(self) -> Self:
        """Reject a `result` or `error` inconsistent with `status`."""

        if self.status is OutcomeStatus.ERROR and not self.error:
            raise ValueError("an error outcome must carry an error message")
        if self.status is not OutcomeStatus.ERROR and self.error is not None:
            raise ValueError(
                f"a {self.status.value} outcome must not carry an error message"
            )
        if self.status is OutcomeStatus.BLOCKED and self.result is not None:
            raise ValueError(
                "a blocked outcome must not carry a result: the tool never ran"
            )
        return self


class ToolCall(_ToolInvocation):
    """One tool invocation, dispatched or refused before dispatch.

    Refusals are recorded because the guard is measured by what it prevented
    and by whether the agent recovered afterwards.
    """

    kind: Literal["tool_call"] = "tool_call"
    index: int = Field(ge=0)
    outcome: ToolOutcome
    # The constraint that stopped the call.
    blocked_by: NonEmptyStr | None = None
    latency_ms: float | None = Field(default=None, ge=0)

    @property
    def blocked(self) -> bool:
        """Return whether the guard refused this call before it ran."""

        return self.outcome.status is OutcomeStatus.BLOCKED

    @property
    def succeeded(self) -> bool:
        """Return whether the tool ran and reported success.

        A blocked or errored call did nothing, so it satisfies no prerequisite
        and produces no side effect.
        """

        return self.outcome.status is OutcomeStatus.OK

    @property
    def result_fields(self) -> Mapping[str, Any]:
        """Return the result as a mapping, or an empty one if it is not an object.

        A tool may return a list or a scalar; a rule reading a named field
        simply does not apply to it.
        """

        return self.outcome.result if isinstance(self.outcome.result, Mapping) else {}

    @model_validator(mode="after")
    def check_blocking_is_recorded_on_both_sides(self) -> Self:
        """Reject a blocked status without `blocked_by`, or the reverse.

        A block with no constraint cannot be attributed. A constraint on a
        call that ran means the guard was consulted and ignored.
        """

        if self.blocked and self.blocked_by is None:
            raise ValueError(
                f"tool call {self.name!r} is blocked but names no constraint. A "
                "refusal that cannot be attributed is not evidence."
            )
        if not self.blocked and self.blocked_by is not None:
            raise ValueError(
                f"tool call {self.name!r} names blocking constraint "
                f"{self.blocked_by!r} but has outcome {self.outcome.status.value!r}. "
                "A blocked call must not have executed."
            )
        return self

    @model_validator(mode="after")
    def check_unparsed_arguments_did_not_succeed(self) -> Self:
        """Reject a successful outcome on a call whose arguments never parsed."""

        if self.raw_arguments is not None and self.outcome.status is OutcomeStatus.OK:
            raise ValueError(
                f"tool call {self.name!r} succeeded despite arguments that could "
                "not be parsed. A call the dispatcher could not read cannot have "
                "run correctly."
            )
        return self


Step = Annotated[LLMCall | ToolCall, Field(discriminator="kind")]
"""One step of a run: an LLM call or a tool call, told apart by `kind`.

Without the discriminator a round trip can silently produce the wrong step
type while an equality test still passes.
"""


class TerminalState(StrEnum):
    """Why the agent loop stopped.

    `ESCALATED` is separate from `COMPLETED` because abstention is a correct
    outcome, and the two must never be aggregated.
    """

    COMPLETED = "completed"
    ESCALATED = "escalated"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
    PROVIDER_ERROR = "provider_error"
    TOOL_ERROR = "tool_error"
    # Three consecutive identical tool calls. Without this a blocked call under
    # the guard can retry without bound.
    ABANDONED = "abandoned"


class Trajectory(Record):
    """One agent run against one instance.

    Readers use the methods below rather than indexing `steps`, so the policy
    layer never depends on how a step records its outcome.
    """

    schema_version: str = SCHEMA_VERSION
    run_id: NonEmptyStr
    instance_id: NonEmptyStr
    # Zero-based. Repeats of an instance are independent.
    repeat_index: int = Field(default=0, ge=0)
    steps: tuple[Step, ...] = ()
    wall_ms: float | None = Field(default=None, ge=0)
    terminal_state: TerminalState
    # What the agent finally said. `None` when it stopped without answering.
    final_answer: str | None = None
    provider_meta: FrozenJsonObject = Field(default_factory=FrozenDict)

    @property
    def llm_calls(self) -> tuple[LLMCall, ...]:
        """Return the LLM calls, in order."""

        return tuple(step for step in self.steps if isinstance(step, LLMCall))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Return the tool calls, in order."""

        return tuple(step for step in self.steps if isinstance(step, ToolCall))

    @property
    def blocked_calls(self) -> tuple[ToolCall, ...]:
        """Return the tool calls the guard refused."""

        return tuple(call for call in self.tool_calls if call.blocked)

    def step_at(self, index: int) -> Step:
        """Return the step at `index`, or raise `IndexError` with a remedy.

        The guard judges a call before making it by appending it and asking
        about its index, so the error says so rather than reading as a bug.
        """

        if index < 0 or index >= len(self.steps):
            raise IndexError(
                f"no step at index {index}: the trajectory has {len(self.steps)} "
                "steps. To judge a call before making it, append it and ask "
                "about its index."
            )
        return self.steps[index]

    def successful_calls_before(
        self, index: int, name: str | None = None
    ) -> tuple[ToolCall, ...]:
        """Return the tool calls before `index` that succeeded, by `name` if given.

        The view a constraint gets of the past: what happened and stuck. A
        blocked or failed call did nothing, so it is left out.
        """

        return tuple(
            step
            for step in self.steps[:index]
            if isinstance(step, ToolCall)
            and step.succeeded
            and (name is None or step.name == name)
        )

    def prompt_at(self, index: int) -> tuple[Message, ...]:
        """Return the request as sent for the LLM call at `index`.

        Earlier deltas interleaved with the replies between them, then this
        call's delta and nothing after. Including this call's own reply would
        ask the provider to continue past its own answer.
        """

        step = self.step_at(index)
        if not isinstance(step, LLMCall):
            raise TypeError(
                f"step {index} is a {step.kind}, not an LLM call. Only LLM calls "
                "have a prompt."
            )

        prompt: list[Message] = []
        for earlier in self.llm_calls:
            if earlier.index > index:
                break
            prompt.extend(earlier.messages_appended)
            if earlier.index < index and earlier.assistant_turn is not None:
                prompt.append(earlier.assistant_turn)
        return tuple(prompt)

    def ends_with_tool(self, name: str) -> bool:
        """Return whether the last step is a successful call to `name`.

        Which tool escalates is domain knowledge, so the `Terminal` rule
        supplies the name and this is the check it drives.
        """

        if not self.steps:
            return False
        last = self.steps[-1]
        return isinstance(last, ToolCall) and last.name == name and last.succeeded

    @model_validator(mode="after")
    def check_step_indices_match_position(self) -> Self:
        """Reject a step whose `index` differs from its position.

        `index` is a second source of truth for the ordering. A violation
        report quotes it, so a mismatch points a reader at the wrong step.
        """

        for position, step in enumerate(self.steps):
            if step.index != position:
                raise ValueError(
                    f"step at position {position} carries index {step.index}. "
                    "Step indices must match their position in the trajectory."
                )
        return self

    @model_validator(mode="after")
    def check_call_ids_are_unique(self) -> Self:
        """Reject two tool calls sharing a `call_id`.

        Duplicate ids make the request-to-dispatch pairing ambiguous. The
        duplicate-call fault profile must show as two calls, not one.
        """

        seen: set[str] = set()
        for call in self.tool_calls:
            if call.call_id in seen:
                raise ValueError(
                    f"call_id {call.call_id!r} appears on more than one tool call. "
                    "Dispatches must be individually identifiable."
                )
            seen.add(call.call_id)
        return self

    @model_validator(mode="after")
    def check_terminal_state_matches_content(self) -> Self:
        """Reject a terminal state the steps and answer contradict.

        Only the domain-blind half: a completed run said something, and
        neither completing nor escalating happens without a step. That an
        escalated run ends in the escalation tool is the `Terminal` rule's job.
        """

        answered = TerminalState.COMPLETED
        if self.terminal_state is answered and self.final_answer is None:
            raise ValueError(
                "a completed trajectory has no final answer. Completion means the "
                "agent answered; an unanswered run is another terminal state."
            )
        if self.terminal_state is not answered and self.final_answer is not None:
            raise ValueError(
                f"a {self.terminal_state.value} trajectory carries a final answer, "
                "but only a completed run answers. Whatever the model said last "
                "is in that step's response_text."
            )
        if (
            self.terminal_state in (answered, TerminalState.ESCALATED)
            and not self.steps
        ):
            raise ValueError(
                f"a trajectory with no steps cannot be {self.terminal_state.value}: "
                "the agent did nothing"
            )
        return self


class UnknownSchemaVersionError(ValueError):
    """A committed trajectory this build does not know how to read."""


_LOADERS: dict[str, Callable[[Mapping[str, Any]], Trajectory]] = {
    SCHEMA_VERSION: Trajectory.model_validate,
}


def load_trajectory(data: str | bytes | Mapping[str, Any]) -> Trajectory:
    """Parse a committed trajectory, dispatching on its `schema_version`.

    When the schema changes, add a converting loader keyed on the old version.
    Never edit a committed trajectory to match new code.
    """

    if isinstance(data, str | bytes):
        data = json.loads(data)
    if not isinstance(data, Mapping):
        raise TypeError(f"expected a JSON object, got {type(data).__name__}")

    version = data.get("schema_version")
    loader = _LOADERS.get(version) if isinstance(version, str) else None
    if loader is None:
        known = ", ".join(sorted(_LOADERS))
        described = repr(version) if version is not None else "no schema_version"
        raise UnknownSchemaVersionError(
            f"cannot read trajectory with {described}; this build knows {known}. "
            "Add a loader for the older version rather than editing the file."
        )
    return loader(data)
