from functools import cached_property
import time
from typing import Literal, Protocol
import copy
import uuid
from pydantic import BaseModel, model_validator
import weaviate

from embedify.embeddings.openai import OpenAIEmbeddingConfig, OpenAIEmbeddingModel
from embedify.types import SupportedVectorDb

EmbeddingModel = Literal["OpenAI", "Cohere", "HuggingFace"]
DEFAULT_BATCH_SIZE = 200


class EmbeddingModelConfig(Protocol):
    model_name: EmbeddingModel
    module_name: str
    api_key: str
    rate_limit: float  # per second

    def model_config(self) -> dict[str, str]:
        pass


class WeaviateOpenAIConfig(OpenAIEmbeddingConfig):
    @property
    def model_name(self) -> EmbeddingModel:
        return "OpenAI"

    @property
    def module_name(self) -> str:
        return f"text2vec-{self.model_name.lower()}"

    def module_config(self) -> dict[str, str]:
        return {
            "model": "ada"
            if self.model == OpenAIEmbeddingModel.ADA
            else self.model.value,
            "dimension": self.dimension,
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
    embedding_model: WeaviateOpenAIConfig

    @model_validator(mode="after")
    def validate_model(self):
        if self.username and not self.password:
            raise ValueError("Password is required when username is provided")
        elif self.api_key and self.username:
            raise ValueError("Only one of api_key or username/password is allowed")

    def auth_config(self) -> weaviate.auth.AuthCredentials:
        if self.api_key:
            return weaviate.auth.AuthApiKey(api_key=self.api_key)
        elif self.username:
            return weaviate.auth.AuthClientPassword(
                username=self.username, password=self.password
            )
        else:
            return None

    @property
    def name(self):
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
        api_key_header = f"X-{embedding_model.model_name}-API-Key"
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
    def current_schema(self):
        return self._client.schema.get(self._config.collection)

    def new_collection_name(self) -> str:
        if self._config.new_collection:
            return self._config.new_collection

        return f"{self._config.collection}-{self.embedding_model.model_name.lower()}-{self.embedding_model.model.value}".replace(
            "-", "_"
        )

    def migrate_schema(self):
        current_schema = self.current_schema
        new_module_config = self.embedding_model.module_config()
        new_schema = copy.deepcopy(current_schema)
        new_class_name = self.new_collection_name()
        new_schema["class"] = new_class_name
        new_schema["moduleConfig"] = {
            self.embedding_model.module_name: new_module_config
        }
        try:
            self._client.schema.create_class(new_schema)
        except weaviate.exceptions.UnexpectedStatusCodeException as e:
            if e.status_code == 422:
                print(f"Schema already exists for {new_class_name}. Reusing")
                pass
            else:
                raise

    def migrate(self, batch_size=None, limit=None):
        self.migrate_schema()
        total_count = 0
        batch_size = min(batch_size or self._config.batch_size, limit)
        obj_inserter = self.object_inserter(batch_size=batch_size)
        obj_inserter.send(None)
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

    def object_inserter(self, batch_size=None):
        def handle_rate_limit(_):
            time_took_to_create_batch = batch_size * (
                self._client.batch.creation_time
                / self._client.batch.recommended_num_objects
            )
            time.sleep(
                max(
                    batch_size / self.rate_limit - time_took_to_create_batch + 1,
                    0,
                )
            )

        new_collection = self.new_collection_name()
        batch_size = batch_size or self._config.batch_size
        self._client.batch.configure(
            batch_size=batch_size,
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

                    batch.add_data_object(
                        class_name=new_collection, data_object=new_obj, uuid=uuid
                    )

    def load_objects(self, batch_size=None) -> list[dict[str, any]]:
        properties = [p["name"] for p in self.current_schema["properties"]]
        query = (
            self._client.query.get(
                class_name=self._config.collection,
                properties=properties,
            )
            .with_additional(["id"])
            .with_limit(batch_size or self._config.batch_size)
        )
        cursor: uuid.UUID | None = None
        while True:
            if cursor:
                result = query.with_after(cursor).do()
            else:
                result = query.do()

            objects = result["data"]["Get"][self._config.collection]
            if not objects:
                break
            cursor = objects[-1]["_additional"]["id"]
            yield objects
