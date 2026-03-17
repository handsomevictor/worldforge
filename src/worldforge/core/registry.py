"""Registry: simple name-based registry for simulation components."""
from __future__ import annotations

from typing import Any


class Registry:
    """
    A general-purpose name → object registry.

    Used internally to look up agent types, environments, and probes by name.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def register(self, name: str, obj: Any) -> None:
        self._store[name] = obj

    def get(self, name: str) -> Any:
        return self._store.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __iter__(self):
        return iter(self._store.items())

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"Registry(keys={list(self._store.keys())})"
