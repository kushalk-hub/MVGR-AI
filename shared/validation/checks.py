"""Small assertion helpers shared by audits, tests and build steps."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


class ValidationError(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def require_columns(columns: Iterable[str], required: Sequence[str], *, where: str) -> None:
    present = set(columns)
    missing = [column for column in required if column not in present]
    if missing:
        raise ValidationError(f"{where}: missing required columns {missing}")


def require_unique(values: Iterable[str], *, where: str) -> int:
    seen: set[str] = set()
    duplicates: set[str] = set()
    count = 0
    for value in values:
        count += 1
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        sample = sorted(duplicates)[:5]
        raise ValidationError(f"{where}: {len(duplicates)} duplicate identifiers, e.g. {sample}")
    return count


def require_disjoint(left: set[str], right: set[str], *, where: str) -> None:
    overlap = left & right
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValidationError(f"{where}: {len(overlap)} overlapping values, e.g. {sample}")
