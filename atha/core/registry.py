from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class VariableKind(str, Enum):
    STATE = "state"
    ALGEBRAIC = "algebraic"
    COMMAND = "command"
    PARAMETER = "parameter"
    OUTPUT = "output"


@dataclass(frozen=True)
class VariableMetadata:
    name: str
    kind: VariableKind
    units: str
    scale: float
    bounds: Tuple[Optional[float], Optional[float]] = (None, None)
    description: str = ""
    owner: Optional[str] = None


@dataclass(frozen=True)
class ResidualMetadata:
    name: str
    units: str
    scale: float
    description: str = ""
    owner: Optional[str] = None


class VariableRegistry:
    """Ordered metadata registry for layout variables."""

    def __init__(self) -> None:
        self._items: List[VariableMetadata] = []
        self._by_name: Dict[str, VariableMetadata] = {}

    def register(
        self,
        name: str,
        kind: VariableKind,
        units: str,
        scale: float,
        bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        description: str = "",
        owner: Optional[str] = None,
    ) -> VariableMetadata:
        if name in self._by_name:
            raise ValueError(f"Variable '{name}' already registered")
        meta = VariableMetadata(
            name=name,
            kind=VariableKind(kind),
            units=units,
            scale=float(scale),
            bounds=bounds,
            description=description,
            owner=owner,
        )
        self._items.append(meta)
        self._by_name[name] = meta
        return meta

    def names(self, kind: Optional[VariableKind] = None) -> List[str]:
        if kind is None:
            return [item.name for item in self._items]
        kind = VariableKind(kind)
        return [item.name for item in self._items if item.kind == kind]

    def index(self, name: str) -> int:
        if name not in self._by_name:
            raise KeyError(name)
        for i, item in enumerate(self._items):
            if item.name == name:
                return i
        raise KeyError(name)

    def __getitem__(self, name: str) -> VariableMetadata:
        return self._by_name[name]

    def __len__(self) -> int:
        return len(self._items)


class ResidualRegistry:
    """Ordered metadata registry for algebraic residuals."""

    def __init__(self) -> None:
        self._items: List[ResidualMetadata] = []
        self._by_name: Dict[str, ResidualMetadata] = {}

    def register(
        self,
        name: str,
        units: str,
        scale: float,
        description: str = "",
        owner: Optional[str] = None,
    ) -> ResidualMetadata:
        if name in self._by_name:
            raise ValueError(f"Residual '{name}' already registered")
        meta = ResidualMetadata(
            name=name,
            units=units,
            scale=float(scale),
            description=description,
            owner=owner,
        )
        self._items.append(meta)
        self._by_name[name] = meta
        return meta

    def names(self) -> List[str]:
        return [item.name for item in self._items]

    def scales(self) -> List[float]:
        return [item.scale for item in self._items]

    def index(self, name: str) -> int:
        if name not in self._by_name:
            raise KeyError(name)
        for i, item in enumerate(self._items):
            if item.name == name:
                return i
        raise KeyError(name)

    def __getitem__(self, name: str) -> ResidualMetadata:
        return self._by_name[name]

    def __len__(self) -> int:
        return len(self._items)
