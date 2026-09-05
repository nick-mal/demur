"""Base class and shared field types for every demur record.

A record is committed evidence: a trajectory, a manifest, an instance. The
settings here must hold for all of them, so they live in one place.
"""

from __future__ import annotations

import hashlib
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

Validated so a truncated or upper-cased digest fails here, not as a
comparison that never matches.
"""

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
"""A non-blank identifier. An empty `run_id` would be a nameless directory."""

TokenCount = Annotated[int, Field(ge=0)]
"""A non-negative token count. `None` elsewhere means not reported."""


def sha256_hex(source: bytes) -> Sha256:
    """Return the SHA-256 hex digest of `source`.

    Identity is over source bytes, never the parsed model: a library change
    must not rewrite the hash of data that did not change. See spec §4.
    """

    return hashlib.sha256(source).hexdigest()


def _refuse(self: object, *_args: object, **_kwargs: object) -> NoReturn:
    """Raise `TypeError` in place of any mutating container method."""

    raise TypeError(
        f"{type(self).__name__} is immutable: scorers derive values from a record "
        "and never annotate it. Build a new object instead of editing in place."
    )


class FrozenList(list[Any]):
    """A `list` whose mutating methods raise `TypeError`.

    `frozen=True` only blocks attribute assignment. Declared sequence fields
    use `tuple`; this covers lists inside JSON values, which must stay
    list-shaped to serialise.
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
    """A `dict` whose mutating methods raise `TypeError`. See `FrozenList`."""

    __setitem__ = _refuse
    __delitem__ = _refuse
    __ior__ = _refuse
    clear = _refuse
    pop = _refuse
    popitem = _refuse
    setdefault = _refuse
    update = _refuse


def freeze(value: Any) -> Any:
    """Return `value` with every list and dict replaced by a frozen one.

    Recursive, so nested tool arguments are immutable at every depth. Both
    frozen types subclass the plain containers, so serialisation, equality
    and content hashes are unaffected.
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
    """Base model for every record: immutable at every depth, strict on fields.

    `frozen=True` blocks assignment; the `Frozen*` types block mutation through
    a field. `extra="forbid"` makes a drifted field name fail at load instead
    of silently dropping a metric. See spec §4.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Return a copy with `update` applied, re-validated.

        Pydantic's own `model_copy` skips every validator, so a copy could
        hold what the original never could. Every invariant here is load bearing.
        """

        copied = super().model_copy(update=dict(update) if update else None, deep=deep)
        if not update:
            return copied
        return type(self).model_validate(copied.__dict__)
