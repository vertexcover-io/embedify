from abc import ABC


class BaseEmbedding(ABC):
    def embed(self, _: str) -> list[float]:
        pass

    def bulk_embed(self, _: list[str]) -> list[list[float]]:
        pass

    async def async_embed(self, _: str) -> list[str]:
        pass

    async def async_bulk_embed(self, _: list[str]) -> list[list[float]]:
        pass

    def dimensions(self) -> int:
        pass
