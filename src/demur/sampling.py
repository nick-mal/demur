"""How a completion was asked for — the decoding parameters, recorded.

Its own module rather than a corner of `record.py`, which holds the frozen base
class and the annotated scalar types every record shares. `Sampling` is neither:
it is a composite model, used by exactly two records that live in two modules
and are deliberately kept independent of one another. Putting it in either of
them would make `manifest.py` and `trajectory.py` depend on each other for one
small value type; putting it in `record.py` would quietly widen what that module
is for. It cannot live in `providers.py`, the seam it most obviously belongs to,
because that module returns an `LLMCall` and would import the trajectory back.

Not bookkeeping. Temperature alone moves an agent between deterministic and
exploratory, and every finding demur publishes is a claim about behaviour *at a
configuration* — a refusal rate, a recovery rate, a cost per accepted outcome.
Comparing a baseline sampled greedily against a candidate sampled at temperature
1 is comparing two different systems, and without this recorded there is nothing
in the artifacts that says so.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from demur.record import FrozenDict, FrozenJsonObject, Record


class Sampling(Record):
    """The decoding parameters a completion was requested with.

    Lives on both `RunManifest` and `LLMCall`, exactly as `model` does, and for
    the same reason: the manifest holds what the run was **configured** with and
    is what baseline comparability is computed over; the call holds what was
    actually **sent** for that call. They agree in an ordinary run. When they
    differ — a retry at a higher temperature after a malformed tool call, a
    fault profile rewriting a request — the difference is the evidence, and a
    schema that could only express one of them would delete it.

    **`None` means we did not specify it**, so the provider substituted its own
    default. That is not the same as any particular value, and it is strictly
    weaker than stating one: provider defaults change under stable model
    aliases, so a run recorded that way is not reproducible even in principle.
    `RunManifest` therefore refuses an unspecified temperature. Per call the
    field stays optional, because a provider adapter that genuinely sends no
    temperature must be able to say so rather than invent one.

    `stop` is the exception to that reading: an empty tuple means no stop
    sequences were supplied, which is what the provider then does too. There is
    no hidden default to be ignorant of.

    Fields are named here when every provider demur targets has the knob and
    means the same thing by it. Everything else — penalties, thinking budgets,
    reasoning effort, provider-specific decoding — goes in `extra`, recorded
    verbatim and never interpreted. The alternative is a normalisation layer
    across providers, which specification §15 rules out and which would have to
    guess at semantics it cannot check.
    """

    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    # The provider's sampling seed — a different RNG from `RunManifest.seed`,
    # which seeds the harness. See the comment there.
    seed: int | None = None
    stop: tuple[str, ...] = ()
    extra: FrozenJsonObject = Field(default_factory=FrozenDict)

    @property
    def specified(self) -> bool:
        """Whether anything was pinned at all.

        A run where this is false is one whose decoding behaviour is whatever
        the provider felt like that day.
        """

        return any(
            value is not None
            for value in (
                self.temperature,
                self.top_p,
                self.top_k,
                self.max_output_tokens,
                self.seed,
            )
        ) or bool(self.stop or self.extra)

    @model_validator(mode="after")
    def check_extra_does_not_shadow_a_named_field(self) -> Self:
        """`extra` is for knobs this model does not name, and only those.

        `extra={"temperature": 0.7}` beside `temperature=None` is two answers
        to one question, and a reader — or a replay, or the baseline hash —
        would take whichever it happened to look at first.
        """

        shadowed = sorted(set(self.extra) & set(type(self).model_fields))
        if shadowed:
            raise ValueError(
                f"sampling `extra` repeats {', '.join(shadowed)}, which this "
                "model names as a field of its own — put the value there. Two "
                "places to read one parameter is one place too many."
            )
        return self
