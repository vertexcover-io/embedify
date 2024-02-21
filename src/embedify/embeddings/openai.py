from enum import Enum
import functools
from typing import Any, Optional

from openai.types import CreateEmbeddingResponse, Embedding

from embedify.embeddings.base import BaseEmbedding
from openai import DEFAULT_MAX_RETRIES, BaseModel, OpenAI


class OpenAIEmbeddingModel(Enum):
    SMALL = "text-embedding-3-small"
    LARGE = "text-embedding-3-large"
    ADA = "text-embeddding-ada-2"


class OpenAIEmbeddingConfig(BaseModel):
    api_key: str
    model: OpenAIEmbeddingModel
    dimension: Optional[int]
    rate_limt: float = 50

    organization: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    max_retries: int = DEFAULT_MAX_RETRIES

    def optional_kwargs(self) -> dict[str, Any]:
        kwargs = {}
        if self.organization:
            kwargs["organization"] = self.organization
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if self.max_retries:
            kwargs["max_retries"] = self.max_retries
        return kwargs


def native_support_guard(func):
    @functools.wraps(func)
    def wrapper(self: "OpenAIEmbedding", *args, **kwargs):
        if self.config.use_native:
            raise NotImplementedError(
                "Vector Db Native support enabled. Use that instead"
            )
        return func(self, *args, **kwargs)

    return wrapper


class OpenAIEmbedding(BaseEmbedding):
    config: OpenAIEmbeddingConfig
    client: OpenAI

    def __init__(self, config: OpenAIEmbeddingConfig):
        self.config = config
        self.client = OpenAI(config.api_key, **config.optional_kwargs())

    def _embed(self, input: str | list[str]) -> list[Embedding]:
        resp: CreateEmbeddingResponse = self.client.embeddings.create(
            input=input,
            model=self.config.model.value,
            dimension=self.config.dimension,
        )
        return resp.data

    @native_support_guard
    def embed(self, text: str) -> list[float]:
        if self.config.use_native:
            raise NotImplementedError(
                "Vector Db Native support enabled. Use that instead"
            )

        resp = self._embed(text)
        return resp[0].embedding

    @native_support_guard
    def bulk_embed(self, text_list: list[str]) -> list[list[float]]:
        resp = self._embed(text_list)
        return [embedding.embedding for embedding in resp]

    async def _async_embed(self, input: str | list[str]) -> list[Embedding]:
        resp: CreateEmbeddingResponse = await self.client.embeddings.create(
            input=input,
            model=self.config.model.value,
            dimension=self.config.dimension,
        )
        return resp.data

    @native_support_guard
    async def async_embed(self, text: str) -> list[str]:
        resp = await self._async_embed(text)
        return resp[0].embedding

    @native_support_guard
    async def async_bulk_embed(self, text_list: list[str]) -> list[list[float]]:
        resp = await self._async_embed(text_list)
        return [embedding.embedding for embedding in resp]

    def dimensions(self) -> int:
        if self.config.dimension:
            return self.config.dimension
        elif self.config.model == OpenAIEmbeddingModel.ADA:
            return 1536
        elif self.config.model == OpenAIEmbeddingModel.LARGE:
            return 3072
        return 1024
