"""GenerationContext: everything the generator sees for one seed."""

from pydantic import BaseModel, ConfigDict

from ragsynth.domain.seed import Seed


class GenerationContext(BaseModel):
    """Assembled generation context (SPEC §6.2).

    ``style_exemplars`` are nearest production queries (register/length
    steering); ``instruction`` is the rendered stratum instruction.
    """

    model_config = ConfigDict(frozen=True)

    seed: Seed
    chunk_texts: tuple[str, ...]
    style_exemplars: tuple[str, ...]
    instruction: str
