"""Decoding parameters, recorded per run and per call.

Own module: `manifest.py` and `trajectory.py` both use it and must not import
each other. See spec §4, Sampling.

Recorded because temperature alone changes what an agent does. A baseline at
temperature 0 and a candidate at 1 are two systems, and without this field the
artifacts cannot show that.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from demur.record import FrozenDict, FrozenJsonObject, Record


class Sampling(Record):
    """The decoding parameters a completion was requested with.

    On `RunManifest` as configured and on `LLMCall` as sent; a difference is
    evidence, not an error. `None` means demur did not specify the knob and the
    provider chose. `stop=()` means no stop sequences, which is what the
    provider then does. Provider-specific knobs go in `extra`, verbatim.
    """

    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    # The provider's sampling seed. `RunManifest.seed` seeds the harness.
    seed: int | None = None
    stop: tuple[str, ...] = ()
    # Knobs this model does not name, recorded verbatim. Normalising them is
    # the compatibility layer spec §15 rules out.
    extra: FrozenJsonObject = Field(default_factory=FrozenDict)

    @property
    def specified(self) -> bool:
        """Return whether any parameter was pinned at all."""

        knobs = (
            name for name in type(self).model_fields if name not in ("stop", "extra")
        )
        return any(getattr(self, name) is not None for name in knobs) or bool(
            self.stop or self.extra
        )

    @model_validator(mode="after")
    def check_extra_does_not_shadow_a_named_field(self) -> Self:
        """Reject an `extra` key that repeats a named field.

        Two places to read one parameter is one too many.
        """

        shadowed = sorted(set(self.extra) & set(type(self).model_fields))
        if shadowed:
            raise ValueError(
                f"sampling `extra` repeats {', '.join(shadowed)}, which is a named "
                "field. Put the value there."
            )
        return self
