from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Iterable, Mapping


def apply_path_overrides(obj: Any, overrides: Mapping[str, Any]) -> Any:
    """Return a copy of ``obj`` with dotted-path overrides applied.

    This is the schema-level perturbation primitive needed for generic sweeps
    and Monte Carlo. It supports dataclasses, dictionaries, lists, and nested
    combinations. Numeric list indexes are allowed for list traversal.
    """

    result = deepcopy(obj)
    for path, value in overrides.items():
        result = _set_path(result, str(path).split("."), value)
    return result


def _set_path(obj: Any, parts: list[str], value: Any) -> Any:
    if not parts:
        return value
    head, tail = parts[0], parts[1:]
    if _is_dataclass_instance(obj):
        current = getattr(obj, head)
        return replace(obj, **{head: _set_path(current, tail, value)})
    if isinstance(obj, dict):
        updated = dict(obj)
        if head not in updated:
            joined = None
            joined_tail: list[str] = []
            for end in range(len(parts), 0, -1):
                candidate = ".".join(parts[:end])
                if candidate in updated:
                    joined = candidate
                    joined_tail = parts[end:]
                    break
            if joined is not None:
                updated[joined] = _set_path(updated[joined], joined_tail, value)
                return updated
        if head not in updated:
            raise KeyError(f"override path segment not found: {head}")
        updated[head] = _set_path(updated[head], tail, value)
        return updated
    if isinstance(obj, list):
        index = int(head)
        updated = list(obj)
        updated[index] = _set_path(updated[index], tail, value)
        return updated
    raise TypeError(f"cannot apply override through object of type {type(obj).__name__}: {head}")


def flatten_overrides(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Normalize list-style override specs into a path->value mapping."""

    overrides: dict[str, Any] = {}
    for item in items:
        path = item.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("override item requires non-empty 'path'")
        if "value" not in item:
            raise ValueError(f"override '{path}' requires 'value'")
        overrides[path] = item["value"]
    return overrides


def _is_dataclass_instance(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type)
