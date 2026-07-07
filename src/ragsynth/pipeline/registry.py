"""Generic registry powering deserialization and the LSP contract tests (SPEC §3.3).

Concrete steps/checks/adapters self-register via decorator at import time.
Registry *instances* live next to the ABC/Protocol they index (e.g. ``STEPS``
in ``pipeline.base``, ``CHECKS`` in ``gate.checks.base``) so ownership and
typing stay precise; this module holds only the machinery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


class RegistryError(KeyError):
    """Raised for unknown or duplicate registry keys, with actionable context."""

    def __str__(self) -> str:
        # KeyError.__str__ repr-quotes its message; keep it human-readable.
        return self.args[0] if self.args else ""


class Registry(Generic[T]):
    """A ``key -> class`` map with actionable errors (SPEC §13).

    Attributes:
        kind: Human-readable name of what is registered (used in errors).
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        """Class decorator registering ``cls`` under ``key``.

        Raises:
            RegistryError: If ``key`` is already registered.
        """

        def decorate(cls: type[T]) -> type[T]:
            if key in self._entries:
                raise RegistryError(
                    f"{self.kind} '{key}' already registered to {self._entries[key].__name__}"
                )
            self._entries[key] = cls
            return cls

        return decorate

    def get(self, key: str) -> type[T]:
        """Look up the class registered under ``key``.

        Raises:
            RegistryError: If ``key`` is unknown; the message lists known keys.
        """
        if key not in self._entries:
            raise RegistryError(f"unknown {self.kind} '{key}'; known: {self.keys()}")
        return self._entries[key]

    def keys(self) -> list[str]:
        """All registered keys, sorted."""
        return sorted(self._entries)
