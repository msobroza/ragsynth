"""Tests for the generic Registry (SPEC §3.3, §13 error contract)."""

import pytest

from ragsynth.pipeline.registry import Registry, RegistryError


class Base:
    pass


@pytest.fixture
def registry() -> Registry[Base]:
    return Registry("widget")


def test_register_and_get(registry: Registry[Base]) -> None:
    @registry.register("alpha")
    class Alpha(Base):
        pass

    assert registry.get("alpha") is Alpha


def test_duplicate_key_rejected(registry: Registry[Base]) -> None:
    @registry.register("alpha")
    class Alpha(Base):
        pass

    with pytest.raises(RegistryError, match="already registered"):

        @registry.register("alpha")
        class AlphaTwo(Base):
            pass


def test_unknown_key_lists_known_keys(registry: Registry[Base]) -> None:
    @registry.register("alpha")
    class Alpha(Base):
        pass

    @registry.register("beta")
    class Beta(Base):
        pass

    with pytest.raises(RegistryError, match=r"unknown widget 'gamma'.*alpha.*beta"):
        registry.get("gamma")


def test_keys_sorted(registry: Registry[Base]) -> None:
    @registry.register("zeta")
    class Zeta(Base):
        pass

    @registry.register("alpha")
    class Alpha(Base):
        pass

    assert registry.keys() == ["alpha", "zeta"]
