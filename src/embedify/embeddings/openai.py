from enum import Enum
from typing import Any, TypeVar

from openai import AsyncOpenAI, BaseModel, OpenAI
from openai._constants import DEFAULT_MAX_RETRIES
from openai.types import CreateEmbeddingResponse, Embedding
from pydantic import model_validator

from embedify.embeddings.base import BaseEmbedding


class OpenAIEmbeddingModel(Enum):
    SMALL = "text-embedding-3-small"
    LARGE = "text-embedding-3-large"
    ADA = "text-embeddding-ada-2"


class OpenAIEmbeddingConfig(BaseModel):
    api_key: str
    model: OpenAIEmbeddingModel
    dimension: int
    rate_limt: float = 50
    organization: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    max_retries: int = DEFAULT_MAX_RETRIES

    @model_validator(mode="before")
    def fix_dimension(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "dimension" not in values:
            if values["model"] == OpenAIEmbeddingModel.ADA:
                values["dimension"] = 1536
            elif values["model"] == OpenAIEmbeddingModel.LARGE:
                values["dimension"] = 3072
            else:
                values["dimension"] = 1024
        return values

    def optional_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.organization:
            kwargs["organization"] = self.organization
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if self.max_retries:
            kwargs["max_retries"] = self.max_retries
        return kwargs


ReturnType = TypeVar("ReturnType")


class OpenAIEmbedding(BaseEmbedding):
    config: OpenAIEmbeddingConfig
    client: OpenAI
    async_client: AsyncOpenAI

    def __init__(self, config: OpenAIEmbeddingConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, **config.optional_kwargs())
        self.async_client = AsyncOpenAI(
            api_key=config.api_key, **config.optional_kwargs()
        )

    def _embed(self, input: str | list[str]) -> list[Embedding]:
        resp: CreateEmbeddingResponse = self.client.embeddings.create(
            input=input,
            model=self.config.model.value,
            dimensions=self.config.dimension,
        )
        return resp.data

    def embed(self, text: str) -> list[float]:
        resp = self._embed(text)
        return resp[0].embedding

    def bulk_embed(self, text_list: list[str]) -> list[list[float]]:
        resp = self._embed(text_list)
        return [embedding.embedding for embedding in resp]

    async def _async_embed(self, input: str | list[str]) -> list[Embedding]:
        resp: CreateEmbeddingResponse = await self.async_client.embeddings.create(
            input=input,
            model=self.config.model.value,
            dimensions=self.config.dimension,
        )
        return resp.data

    async def async_embed(self, text: str) -> list[float]:
        resp = await self._async_embed(text)
        return resp[0].embedding

    async def async_bulk_embed(self, text_list: list[str]) -> list[list[float]]:
        resp = await self._async_embed(text_list)
        return [embedding.embedding for embedding in resp]
