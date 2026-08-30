"""The canonical record of what an agent did.

A trajectory is the only artifact everything downstream reads: constraints are
evaluated against it, scorers score it, the regression gate diffs runs of it,
and `make reproduce-published-results` recomputes the headline tables from
committed copies of it with no provider calls. That last guarantee is why every
model here round-trips losslessly through JSON, and why `schema_version` plus
`load_trajectory()` exist — a run committed today has to stay readable after
the code that wrote it has moved on.

Run-level identity — model, treatment, dataset version, prompt hashes — lives
on the `RunManifest`, not here. A trajectory says what happened in one run
against one instance; the manifest says what the run was.

**Prompts are recorded as deltas.** Each `LLMCall` stores the messages that
arrived *before* it — new user or tool turns — and never its own reply, which
lives in `response_text` and `tool_calls_requested` and is reconstituted by
`LLMCall.assistant_turn`. One source of truth per turn: storing the reply in
both places would let a replay and the invalid-attempt count read different
things from the same step and neither would look wrong.

`Trajectory.prompt_at()` rebuilds the exact request for any call by
interleaving earlier deltas with the assistant turns between them. Storing
every request verbatim would repeat the whole conversation at every step and
grow quadratically in committed bytes for no added information. The
reconstruction is exact **only because the agent loop appends**: if it ever
rewrites history — compaction, context editing, a retry that drops a turn — a
delta record stops being faithful and this schema needs a new version rather
than a quiet reinterpretation.

The models are domain-blind by construction. A `ToolCall` knows a tool's name
and the arguments it was given; it does not know that some deployment calls a
tool `run_query` or that the argument happens to be SQL.
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

SCHEMA_VERSION = "1"
"""Bump when a change makes an older committed trajectory unreadable as-is."""


class Usage(Record):
    """Token accounting for one LLM call.

    **`None` is not zero.** A provider that does not report a count leaves the
    field `None`; a provider that reports zero says zero. Collapsing the two
    would turn "we don't know what this cost" into "this was free", and the
    cost-per-accepted-outcome figure would quietly absorb the difference.

    **Which fields overlap.** `input_tokens` and `cached_input_tokens` are
    disjoint — a prompt token was either read from cache or it was not, and the
    prompt total is their sum. `reasoning_tokens` is the exception: it is the
    part of `output_tokens` the model spent thinking, reported for visibility
    and already counted there, so adding it again double-counts.

    Providers disagree about that, and normalising is `providers.py`'s job, not
    the reader's. Anthropic reports the prompt buckets separately, which maps
    across directly. OpenAI-compatible providers — including demur's own fault
    proxy — nest `cached_tokens` inside `prompt_tokens` and `reasoning_tokens`
    inside `completion_tokens`, so an adapter must subtract the cached count
    out rather than copy both fields across. An adapter for a provider with no
    cache at all writes `cached_input_tokens=0`, which is the normalisation the
    validator below insists on.

    No cost field, by specification §4: cost is computed at reporting time from
    a versioned price table, so a run recorded in one month does not carry that
    month's prices into a later table. Field names follow the OpenTelemetry
    `gen_ai.usage.*` convention, since the same numbers become span attributes.
    """

    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    cached_input_tokens: TokenCount | None = None
    reasoning_tokens: TokenCount | None = None

    @property
    def reported(self) -> bool:
        """Whether the provider said anything about this call at all.

        Any field being set counts. Keying this on input or output alone would
        let `Usage(cached_input_tokens=7)` through as "nothing reported", which
        is the exact half-report the validator below exists to catch.
        """

        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.cached_input_tokens,
                self.reasoning_tokens,
            )
        )

    @property
    def total_tokens(self) -> int | None:
        """Prompt plus output, or `None` if the provider reported nothing.

        Reasoning is inside `output_tokens` already and is not added again.
        """

        if not self.reported:
            return None
        parts = (self.input_tokens, self.cached_input_tokens, self.output_tokens)
        if any(part is None for part in parts):
            return None
        return sum(part for part in parts if part is not None)

    @model_validator(mode="after")
    def check_reasoning_fits_inside_output(self) -> Self:
        if self.reasoning_tokens is None or self.output_tokens is None:
            return self
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError(
                f"reasoning_tokens ({self.reasoning_tokens}) exceeds output_tokens "
                f"({self.output_tokens}) — reasoning is the thinking part of the "
                "output, not a bucket beside it. A provider adapter reporting "
                "them separately must add reasoning into output."
            )
        return self

    @model_validator(mode="after")
    def check_usage_is_all_or_nothing(self) -> Self:
        """Reported usage must include the cache bucket.

        Half-reported usage is the failure that costs nothing to create and
        everything to notice: cost accounting reads `total_tokens`, and an
        adapter that forgot the cache bucket would make it `None` for every
        call while every other field looked healthy. Either the provider said
        nothing — all four `None` — or the adapter normalised it, which means
        writing `0` where the provider has no cache.
        """

        if not self.reported:
            return self
        missing = [
            name
            for name in ("input_tokens", "output_tokens", "cached_input_tokens")
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                f"usage reports some counts but leaves {', '.join(missing)} "
                "unreported — normalise in the provider adapter and write 0 "
                "where the provider has no such bucket. `None` means the "
                "provider said nothing at all, and it must mean that for every "
                "field or none of them."
            )
        return self


class ToolRequest(Record):
    """A tool call as the model asked for it, before anything dispatched it.

    Separate from `ToolCall`, which is what the dispatcher did about it. The
    invalid-attempt rate is defined as model *intent* — unaffected by whether
    the guard let the call through — so the request has to be recorded in its
    own right and paired with its dispatch by `call_id`.
    """

    # Required, not optional: a request that cannot be paired with its
    # dispatch is not evidence of intent. Providers that emit no id leave the
    # adapter to synthesise one, the same duty it has for token counts.
    call_id: NonEmptyStr
    name: NonEmptyStr
    arguments: FrozenJsonObject = Field(default_factory=FrozenDict)
    # The argument text as the model emitted it, recorded **only** when it
    # could not be parsed. Fault profile 3 injects exactly this, and flattening
    # it into an empty `arguments` would destroy the evidence that the run is
    # meant to produce.
    raw_arguments: str | None = None

    @model_validator(mode="after")
    def check_unparsed_arguments_are_not_also_parsed(self) -> Self:
        if self.raw_arguments is not None and self.arguments:
            raise ValueError(
                f"tool request {self.name!r} carries both parsed arguments and "
                "raw_arguments — raw_arguments records text that could not be "
                "parsed, so having both means one of them is a fiction"
            )
        return self


class Message(Record):
    """One entry of the conversation as it was actually sent.

    Faithful enough to replay: an assistant turn that only calls tools has no
    content and carries the calls it asked for, which is what every
    OpenAI-compatible provider emits and what a committed trajectory needs if
    it is ever to be fed back into one.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    # Assistant turns only: the calls this turn asked for.
    tool_calls: tuple[ToolRequest, ...] = ()
    # Tool turns only: which `ToolRequest.call_id` this result answers.
    tool_call_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def check_shape_matches_role(self) -> Self:
        if self.role != "assistant" and self.tool_calls:
            raise ValueError(f"a {self.role} message cannot request tool calls")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError(
                "a tool message must name the call it answers — an unattributed "
                "result cannot be paired with the request that produced it"
            )
        if not (self.content or "").strip() and not self.tool_calls:
            raise ValueError(
                f"a {self.role} message has neither content nor tool calls — "
                "an empty turn is a recording bug, not something a model sent. "
                "Blank content counts as none: a tool with nothing to say still "
                "returns a representation of nothing, such as an empty list."
            )
        return self


class LLMCall(Record):
    """One completion request and what came back.

    `messages_appended` holds what arrived *before* this call — the new user or
    tool turns since the previous one — and never the reply this call produced.
    The reply is `response_text` and `tool_calls_requested`, and
    `assistant_turn` renders it back into a `Message` when a prompt is
    rebuilt. Prompts are recorded rather than regenerated from templates
    because the policy prompt is the experimental manipulation; the proof that
    it was identical across treatments is `system_prompt_sha256` on the
    manifest, which is a property of the run rather than of any one step.
    """

    kind: Literal["llm_call"] = "llm_call"
    index: int = Field(ge=0)
    model: NonEmptyStr
    messages_appended: tuple[Message, ...]
    response_text: str | None = None
    # What the model asked to call. Each request that reached the dispatcher
    # also appears as its own `ToolCall` step, paired on `call_id` — including
    # one the guard blocked and one whose arguments failed to parse.
    tool_calls_requested: tuple[ToolRequest, ...] = ()
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float | None = Field(default=None, ge=0)
    provider_request_id: str | None = None

    @property
    def assistant_turn(self) -> Message | None:
        """This call's reply, as the message that entered the conversation.

        Derived rather than stored, so a replay and the invalid-attempt count
        cannot read different things from the same step. `None` when the call
        produced nothing — a provider error contributes no turn.
        """

        if not (self.response_text or "").strip() and not self.tool_calls_requested:
            return None
        return Message(
            role="assistant",
            content=self.response_text,
            tool_calls=self.tool_calls_requested,
        )

    @model_validator(mode="after")
    def check_the_reply_is_not_also_in_the_delta(self) -> Self:
        """A delta must not end with an assistant turn.

        That is where this call's own reply would land if someone appended it,
        and `prompt_at()` would then feed the model its own answer back. Other
        assistant messages are legitimate — few-shot exemplars arrive as
        user/assistant pairs — but an exemplar is never the last thing before
        the request, since the request itself follows it.
        """

        if self.messages_appended and self.messages_appended[-1].role == "assistant":
            raise ValueError(
                "the messages before an LLM call end with an assistant turn — "
                "a call's own reply belongs in response_text and "
                "tool_calls_requested, not in the delta it was sent with"
            )
        return self


class OutcomeStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    # Refused by the dispatch guard before execution. Not an error: the tool
    # never ran, and counting it as a failure would score the guard doing its
    # job as the agent doing something wrong.
    BLOCKED = "blocked"


class ToolOutcome(Record):
    """What came of a tool call — including a call that never ran."""

    status: OutcomeStatus
    result: FrozenJson = None
    error: str | None = None

    @model_validator(mode="after")
    def check_fields_match_status(self) -> Self:
        if self.status is OutcomeStatus.ERROR and not self.error:
            raise ValueError("an error outcome must carry an error message")
        if self.status is not OutcomeStatus.ERROR and self.error is not None:
            raise ValueError(
                f"a {self.status.value} outcome must not carry an error message"
            )
        if self.status is OutcomeStatus.BLOCKED and self.result is not None:
            raise ValueError(
                "a blocked outcome must not carry a result — the tool never ran"
            )
        return self


class ToolCall(Record):
    """One tool invocation — dispatched, or refused before dispatch.

    `blocked_by` names the constraint that stopped the call. It is the whole
    point of recording refusals: the dispatch guard is measured by what it
    prevented and by whether the agent recovered afterwards, and neither is
    visible if a blocked call leaves no trace.
    """

    kind: Literal["tool_call"] = "tool_call"
    index: int = Field(ge=0)
    name: NonEmptyStr
    arguments: FrozenJsonObject = Field(default_factory=FrozenDict)
    # Set only when the model's arguments could not be parsed. See `ToolRequest`.
    raw_arguments: str | None = None
    outcome: ToolOutcome
    blocked_by: NonEmptyStr | None = None
    # Pairs this dispatch with the `ToolRequest` that asked for it. Required
    # for the same reason it is on the request: an unpairable dispatch cannot
    # be attributed to the intent that produced it.
    call_id: NonEmptyStr
    latency_ms: float | None = Field(default=None, ge=0)

    @property
    def blocked(self) -> bool:
        return self.outcome.status is OutcomeStatus.BLOCKED

    @model_validator(mode="after")
    def check_blocking_is_recorded_on_both_sides(self) -> Self:
        """`blocked` status and `blocked_by` travel together, or not at all.

        One without the other is a recording bug that reads as a real result:
        a blocked call with no constraint named cannot be attributed, and a
        named constraint on a call that ran means the guard was consulted and
        then ignored.
        """

        if self.blocked and self.blocked_by is None:
            raise ValueError(
                f"tool call {self.name!r} is blocked but names no constraint — "
                "a refusal that cannot be attributed is not evidence"
            )
        if not self.blocked and self.blocked_by is not None:
            raise ValueError(
                f"tool call {self.name!r} names blocking constraint "
                f"{self.blocked_by!r} but has outcome {self.outcome.status.value!r} "
                "— a blocked call must not have executed"
            )
        return self

    @model_validator(mode="after")
    def check_unparsed_arguments_did_not_succeed(self) -> Self:
        if self.raw_arguments is None:
            return self
        if self.arguments:
            raise ValueError(
                f"tool call {self.name!r} carries both parsed arguments and "
                "raw_arguments — raw_arguments records text that could not be "
                "parsed, so having both means one of them is a fiction"
            )
        if self.outcome.status is OutcomeStatus.OK:
            raise ValueError(
                f"tool call {self.name!r} succeeded despite arguments that could "
                "not be parsed — a call the dispatcher could not read cannot "
                "have run correctly"
            )
        return self


Step = Annotated[LLMCall | ToolCall, Field(discriminator="kind")]
"""A step is one or the other; `kind` decides which on the way back from JSON.

Without the discriminator a round trip can silently produce the wrong step type
— a union tries each member in turn and takes the first that fits — and an
equality test written against the same models would still pass.
"""


class TerminalState(StrEnum):
    """Why the agent loop stopped.

    Distinct from whether the answer was any good — that is the scorers' job.
    `ESCALATED` is separate from `COMPLETED` because abstention is a correct
    outcome, not a failure, and the two must never be aggregated together.
    """

    COMPLETED = "completed"
    ESCALATED = "escalated"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
    PROVIDER_ERROR = "provider_error"
    TOOL_ERROR = "tool_error"
    # Three consecutive identical tool calls. Without loop detection, a blocked
    # call under the dispatch guard can retry unboundedly.
    ABANDONED = "abandoned"


class Trajectory(Record):
    """One agent run against one instance."""

    schema_version: str = SCHEMA_VERSION
    run_id: NonEmptyStr
    instance_id: NonEmptyStr
    # Which repeat this is, zero-based. Repeats of an instance are independent.
    repeat_index: int = Field(default=0, ge=0)
    steps: tuple[Step, ...] = ()
    wall_ms: float | None = Field(default=None, ge=0)
    terminal_state: TerminalState
    # What the agent finally said. `None` when it stopped without answering.
    final_answer: str | None = None
    provider_meta: FrozenJsonObject = Field(default_factory=FrozenDict)

    @property
    def llm_calls(self) -> tuple[LLMCall, ...]:
        return tuple(step for step in self.steps if isinstance(step, LLMCall))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(step for step in self.steps if isinstance(step, ToolCall))

    @property
    def blocked_calls(self) -> tuple[ToolCall, ...]:
        return tuple(call for call in self.tool_calls if call.blocked)

    def prefix(self, upto: int) -> tuple[Step, ...]:
        """The steps before index `upto`.

        Constraints are checked against what had already happened when a step
        was about to be taken, so this is the view they get.
        """

        if upto < 0:
            raise IndexError(f"step index must not be negative, got {upto}")
        return self.steps[:upto]

    def prompt_at(self, index: int) -> tuple[Message, ...]:
        """The request as it was sent for the LLM call at step `index`.

        Earlier deltas interleaved with the assistant turns between them, then
        this call's delta — and nothing after it. The boundary matters: include
        this call's own reply and the reconstruction asks a provider to
        continue past its own answer, which is not the request that was made.

        Exact under the append-only loop described in the module docstring, and
        the reason the record stays linear in size without losing what was sent.
        """

        if index < 0:
            raise IndexError(f"step index must not be negative, got {index}")
        step = self.steps[index]
        if not isinstance(step, LLMCall):
            raise TypeError(
                f"step {index} is a {step.kind}, not an LLM call — only LLM "
                "calls have a prompt"
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
        """Whether the last step is a successful call to `name`.

        The library cannot check that an `ESCALATED` run really escalated,
        because which tool escalates is domain knowledge: it comes from the
        constraint set's `Terminal` rule, not from a name hard-coded here. This
        is the hook that check uses.
        """

        if not self.steps:
            return False
        last = self.steps[-1]
        return (
            isinstance(last, ToolCall)
            and last.name == name
            and last.outcome.status is OutcomeStatus.OK
        )

    @model_validator(mode="after")
    def check_step_indices_match_position(self) -> Self:
        """`index` must agree with position in `steps`.

        The field is a second source of truth for the ordering, which is what
        makes it worth checking: a constraint reads position and a violation
        report quotes `index`, so a trajectory where they disagree would point
        a reader at the wrong step.
        """

        for position, step in enumerate(self.steps):
            if step.index != position:
                raise ValueError(
                    f"step at position {position} carries index {step.index} — "
                    "step indices must match their position in the trajectory"
                )
        return self

    @model_validator(mode="after")
    def check_call_ids_are_unique(self) -> Self:
        """Two dispatches cannot share a `call_id`.

        Pairing a request with its dispatch is how model intent is separated
        from what the guard allowed. Duplicate ids make that pairing ambiguous,
        and a duplicated id is also what the duplicate-side-effect fault
        profile produces — which must show up as two calls, not one collapsed.
        """

        seen: set[str] = set()
        for call in self.tool_calls:
            if call.call_id is None:
                continue
            if call.call_id in seen:
                raise ValueError(
                    f"call_id {call.call_id!r} appears on more than one tool "
                    "call — dispatches must be individually identifiable"
                )
            seen.add(call.call_id)
        return self

    @model_validator(mode="after")
    def check_terminal_state_matches_content(self) -> Self:
        """A terminal state has to be consistent with what the run contains.

        Only the domain-blind half is checkable here: a run that completed must
        have said something, and neither completing nor escalating is possible
        without taking a step. That an `ESCALATED` run really ends in the
        escalation tool is a `Terminal` constraint, since the tool's name is
        domain knowledge — see `ends_with_tool()`.
        """

        answered = TerminalState.COMPLETED
        if self.terminal_state is answered and self.final_answer is None:
            raise ValueError(
                "a completed trajectory has no final answer — completion means "
                "the agent answered, and an unanswered run is one of the other "
                "terminal states"
            )
        if self.terminal_state is not answered and self.final_answer is not None:
            raise ValueError(
                f"a {self.terminal_state.value} trajectory carries a final "
                "answer — a run that escalated, ran out of steps or failed "
                "stopped without answering. Whatever the model said last is in "
                "that step's response_text; a final answer is what the run "
                "returned, and these two must not be conflated"
            )
        if (
            self.terminal_state in (answered, TerminalState.ESCALATED)
            and not self.steps
        ):
            raise ValueError(
                f"a trajectory with no steps cannot be {self.terminal_state.value} "
                "— the agent did nothing"
            )
        return self


class UnknownSchemaVersionError(ValueError):
    """A committed trajectory this build does not know how to read."""


_LOADERS: dict[str, Callable[[Mapping[str, Any]], Trajectory]] = {
    SCHEMA_VERSION: Trajectory.model_validate,
}


def load_trajectory(data: str | bytes | Mapping[str, Any]) -> Trajectory:
    """Read a committed trajectory, dispatching on its `schema_version`.

    The sanctioned way in, for JSON text or an already-parsed mapping. Runs
    committed today must stay readable once the models have moved on, so a
    version this build cannot read fails with a message that says so rather
    than with a pile of field errors that look like corrupt data.

    When the schema changes, add a converting loader here keyed on the old
    version. Never edit a committed trajectory to match new code.
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
            "Add a loader for the older version rather than editing the "
            "committed file — it is the evidence, and the code is not."
        )
    return loader(data)
