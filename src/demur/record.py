"""The base class every demur record derives from, and the types they share.

A record is a piece of committed evidence — a trajectory, a run manifest, an
instance. The settings here are decisions rather than defaults, and they belong
in one place because they have to hold for all of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, NoReturn, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
"""A lowercase hex SHA-256 digest.

Content addressing is only worth anything if the addresses are well formed, and
a truncated or upper-cased digest copied by hand would otherwise sit in a
manifest looking plausible until a comparison silently failed to match.
"""

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
"""An identifier that has to actually identify something.

A bare `str` accepts `""`, and an empty `run_id` becomes a `runs/` directory
with no name — the kind of thing that surfaces as a confusing filesystem error
hours into a run rather than as a validation failure at the start of it.
"""

TokenCount = Annotated[int, Field(ge=0)]
"""A non-negative token count. `None` elsewhere means *not reported*."""


def _refuse(self: object, *_args: object, **_kwargs: object) -> NoReturn:
    raise TypeError(
        f"{type(self).__name__} is immutable — this is part of a committed "
        "record. Scorers derive values from a trajectory; they never annotate "
        "it. Build a new object from the old one instead of editing in place."
    )


class FrozenList(list[Any]):
    """A list that refuses to change.

    `frozen=True` on a pydantic model only blocks attribute assignment: the
    list behind a field is still an ordinary list, and `traj.steps.append(...)`
    would edit the record in place without tripping anything. Declared fields
    use `tuple` instead; this exists for the sequences *inside* JSON values,
    where the declared type has to stay list-shaped for serialisation.
    """

    __setitem__ = _refuse
    __delitem__ = _refuse
    __iadd__ = _refuse
    __imul__ = _refuse
    append = _refuse
    clear = _refuse
    extend = _refuse
    insert = _refuse
    pop = _refuse
    remove = _refuse
    reverse = _refuse
    sort = _refuse


class FrozenDict(dict[Any, Any]):
    """A mapping that refuses to change. See `FrozenList`."""

    __setitem__ = _refuse
    __delitem__ = _refuse
    __ior__ = _refuse
    clear = _refuse
    pop = _refuse
    popitem = _refuse
    setdefault = _refuse
    update = _refuse


def freeze(value: Any) -> Any:
    """Recursively replace the containers in a JSON value with frozen ones.

    Depth matters: a tool call's `arguments` can nest, and a guard that stops
    at the top level would still let `arguments["filters"].append(...)` rewrite
    what the agent did after the fact.

    `FrozenList` and `FrozenDict` subclass `list` and `dict`, so pydantic
    serialises them natively and they still compare equal to the plain
    containers a JSON payload parses into — round trips and content hashes are
    unaffected.
    """

    if isinstance(value, dict):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return FrozenList(freeze(item) for item in value)
    return value


FrozenJson = Annotated[JsonValue, AfterValidator(freeze)]
"""Arbitrary JSON that cannot be edited once it is part of a record."""

FrozenJsonObject = Annotated[dict[str, FrozenJson], AfterValidator(FrozenDict)]
"""A JSON object field, frozen at every level."""

FrozenStrMap = Annotated[dict[str, str], AfterValidator(FrozenDict)]
"""A string-to-string mapping field, frozen."""


class Record(BaseModel):
    """Immutable — genuinely, not just at the top level.

    This is the first of the two invariants in specification §4: trajectories
    are immutable once written, and scorers derive values rather than annotate.
    `frozen=True` blocks assignment to a field; the tuple and `Frozen*` types
    above block the mutation *through* a field that assignment alone leaves
    open. Both halves are needed — the second is the one that fails quietly.

    `extra="forbid"`: records are read back months after they were written. A
    field name that drifts has to fail loudly at load rather than be silently
    dropped and take a metric with it.

    Build records through validation. `model_copy(update=...)` is pydantic's
    one route around every validator on this class, so it is overridden below
    to re-validate rather than left as an unguarded back door.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Copy with changes, re-validating the result.

        Pydantic's own `model_copy` applies `update` without running a single
        validator, which on these models means a copy can hold what the
        original could never have been: a completed trajectory with no answer,
        steps whose indices no longer match their positions, an ambiguous
        instance expecting a confident answer. Every invariant here is load
        bearing for a measurement, so the copy goes back through validation.
        """

        copied = super().model_copy(update=dict(update) if update else None, deep=deep)
        if not update:
            return copied
        return type(self).model_validate(copied.__dict__)
