from typing import Any, Type
from embedify.types import SupportedVectorDb, VectorDbMigrateClient
from embedify.vectordb.weaviate import (
    WeaviateConfig,
    WeaviateMigrateClientWithNativeEmbedding,
)

VectorDbConfig = WeaviateConfig


def load_config(config_data: dict[str, Any]) -> VectorDbConfig:
    return WeaviateConfig(**config_data)


__vector_db_migrate_client__: dict[str, Type[VectorDbMigrateClient]] = {
    SupportedVectorDb.Weaviate.name: WeaviateMigrateClientWithNativeEmbedding
}


def get_migration_client(config: VectorDbConfig) -> VectorDbMigrateClient:
    client_cls = __vector_db_migrate_client__.get(config.name)
    if not client_cls:
        raise ValueError(f"Unsupported Vector DB: {config.name}")
    return client_cls(config)
