import abc
from functools import cached_property
import time
from typing import Any, Generator, Literal, cast
import copy
import uuid
from pydantic import BaseModel, Field, model_validator
import weaviate

from embedify.embeddings.openai import OpenAIEmbeddingConfig, OpenAIEmbeddingModel
from embedify.types import DEFAULT_LIMIT, SupportedVectorDb

EmbeddingModelType = Literal["OpenAI"] | Literal["Cohere"]


class EmbeddingModelConfig(BaseModel, abc.ABC):
    api_key: str
    rate_limit: float  # per second

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        ...

    @abc.abstractmethod
    def module_config(self) -> dict[str, Any]:
        ...

    @property
    @abc.abstractmethod
    def module_name(self) -> str:
        ...


class WeaviateOpenAIConfig(OpenAIEmbeddingConfig, EmbeddingModelConfig):
    platform: Literal["OpenAI"]

    @property
    def model_name(self) -> str:
        return self.model.value

    def module_config(self) -> dict[str, Any]:
        return {
            "model": "ada"
            if self.model == OpenAIEmbeddingModel.ADA
            else self.model.value,
            "dimension": self.dimension,
        }

    @property
    def module_name(self) -> str:
        return f"text2vec-{self.platform.lower()}"


class WeaviateCohereConfig(EmbeddingModelConfig):
    platform: Literal["Cohere"]

    @property
    def module_name(self) -> str:
        return f"text2vec-{self.platform.lower()}"

    @property
    def model_name(self) -> str:
        return ""

    def module_config(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "dimension": 1024,
        }


class WeaviateConfig(BaseModel):
    cluster_url: str
    collection: str
    new_collection: str | None = None
    api_key: str | None = None
    username: str | None = None
    password: str | None = None
    batch_size: int = 200
    use_native: bool = True
    embedding_model: WeaviateOpenAIConfig | WeaviateCohereConfig = Field(
        ..., discriminator="platform"
    )

    @model_validator(mode="after")  # type: ignore
    def validate_model(self) -> "WeaviateConfig":
        if self.username and not self.password:
            raise ValueError("Password is required when username is provided")
        elif self.api_key and self.username:
            raise ValueError("Only one of api_key or username/password is allowed")

        return self

    def auth_config(self) -> weaviate.auth.AuthCredentials | None:
        if self.api_key:
            return weaviate.auth.AuthApiKey(api_key=self.api_key)
        elif self.username and self.password:
            return weaviate.auth.AuthClientPassword(
                username=self.username, password=self.password
            )
        else:
            return None

    @property
    def name(self) -> str:
        return SupportedVectorDb.Weaviate.name


class WeaviateMigrateClientWithNativeEmbedding:
    _config: WeaviateConfig
    _client: weaviate.Client
    rate_limit: float

    def __init__(self, config: WeaviateConfig):
        self._config = config
        embedding_model = config.embedding_model
        self.rate_limit = embedding_model.rate_limit

        additional_headers = {}
        api_key_header = f"X-{embedding_model.platform}-API-Key"
        additional_headers = {api_key_header: embedding_model.api_key}

        self._client = weaviate.Client(
            auth_client_secret=self._config.auth_config(),
            url=self._config.cluster_url,
            additional_headers=additional_headers,
        )

    @cached_property
    def embedding_model(self):
        return self._config.embedding_model

    @cached_property
    def current_schema(self) -> dict[str, Any]:
        return self._client.schema.get(self._config.collection)  # type: ignore

    def new_collection_name(self) -> str:
        if self._config.new_collection:
            return self._config.new_collection

        return f"{self._config.collection}-{self.embedding_model.platform.lower()}-{self.embedding_model.model_name}".replace(
            "-", "_"
        )

    def migrate_schema(self):
        current_schema = self.current_schema
        new_module_config = self.embedding_model.module_config()
        new_schema: dict[str, Any] = copy.deepcopy(current_schema)
        new_class_name = self.new_collection_name()
        new_schema["class"] = new_class_name
        new_schema["moduleConfig"] = {
            self.embedding_model.module_name: new_module_config
        }
        try:
            self._client.schema.create({"classes": [new_schema]})  # type: ignore
        except weaviate.exceptions.UnexpectedStatusCodeException as e:
            if e.status_code == 422:
                print(f"Schema already exists for {new_class_name}. Reusing")
                pass
            else:
                raise

    def migrate(self, limit: int = DEFAULT_LIMIT, batch_size: int | None = None):
        self.migrate_schema()
        total_count = 0
        batch_size = min(batch_size or self._config.batch_size, limit)
        obj_inserter = self.object_inserter(batch_size=batch_size)
        next(obj_inserter)
        for batch in self.load_objects(batch_size=batch_size):
            end = min(total_count + len(batch), limit)
            objects = batch[: end - total_count]
            obj_inserter.send(objects)
            total_count += len(objects)
            if total_count >= limit:
                break
        try:
            obj_inserter.close()
        except GeneratorExit:
            pass

    def object_inserter(
        self, batch_size: int | None = None
    ) -> Generator[None, list[dict[str, Any]], None]:
        up_batch_size: int = batch_size or self._config.batch_size

        def handle_rate_limit(_):
            time_took_to_create_batch: float = cast(
                float,
                up_batch_size
                * (  # type: ignore
                    self._client.batch.creation_time
                    / self._client.batch.recommended_num_objects
                ),
            )
            time.sleep(
                max(
                    up_batch_size / self.rate_limit - time_took_to_create_batch + 1,
                    0,
                )
            )

        new_collection = self.new_collection_name()

        self._client.batch.configure(  # type: ignore
            batch_size=up_batch_size,
            dynamic=True,
            callback=handle_rate_limit,
        )
        with self._client.batch as batch:
            while True:
                objects = yield
                for obj in objects:
                    new_obj = copy.deepcopy(obj)
                    uuid = new_obj["_additional"]["id"]
                    del new_obj["_additional"]

                    batch.add_data_object(  # type: ignore
                        class_name=new_collection, data_object=new_obj, uuid=uuid
                    )

    def load_objects(
        self, batch_size: int | None = None
    ) -> Generator[list[dict[str, Any]], None, None]:
        properties = [p["name"] for p in self.current_schema["properties"]]
        query = (
            self._client.query.get(  # type: ignore
                class_name=self._config.collection,
                properties=properties,
            )
            .with_additional(["id"])
            .with_limit(batch_size or self._config.batch_size)
        )
        cursor: uuid.UUID | None = None
        while True:
            if cursor:
                result = query.with_after(cursor).do()  # type: ignore
            else:
                result = query.do()  # type: ignore

            objects = cast(
                list[dict[str, Any]], result["data"]["Get"][self._config.collection]
            )
            if not objects:
                break
            cursor = objects[-1]["_additional"]["id"]
            yield objects
