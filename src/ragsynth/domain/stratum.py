"""Stratum: Hamel-style dimension tuple identifying a query sub-population."""

from pydantic import BaseModel, ConfigDict


class Stratum(BaseModel):
    """A point in the stratification grid (SPEC §4, §6.1).

    v1 uses a single dimension ``{"query_type": "factoid" | "howto" | "keyword"}``;
    later versions add persona, difficulty, conversational strata (SPEC R3, R8).
    """

    model_config = ConfigDict(frozen=True)

    dims: dict[str, str]

    def key(self) -> str:
        """Canonical string form, sorted by dimension name.

        Returns:
            ``"persona=broker|query_type=factoid"``-style key; stable across
            dict insertion orders so it is safe as a grouping/reporting key.
        """
        return "|".join(f"{k}={v}" for k, v in sorted(self.dims.items()))
