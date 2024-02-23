from abc import ABC
import abc


class BaseEmbedding(ABC):
    @abc.abstractmethod
    def embed(self, text: str) -> list[float]:
        pass

    @abc.abstractmethod
    def bulk_embed(self, text_list: list[str]) -> list[list[float]]:
        pass

    @abc.abstractmethod
    async def async_embed(self, text: str) -> list[float]:
        pass

    @abc.abstractmethod
    async def async_bulk_embed(self, text_list: list[str]) -> list[list[float]]:
        pass

    @abc.abstractmethod
    def dimensions(self) -> int:
        pass
