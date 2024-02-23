import enum
import importlib
from typing import Protocol, Type

DEFAULT_LIMIT = 2**64 - 1


class SupportedVectorDb(enum.Enum):
    Weaviate = "weaviate"


class SupportedEmbedding(enum.Enum):
    OpenAI = "openai"


__vector_db_pkg_map__ = {
    SupportedVectorDb.Weaviate.name: "weaviate",
}


def get_installed_vector_dbs() -> Type[enum.Enum]:
    installed_vector_dbs = {}
    for key, pkg in __vector_db_pkg_map__.items():
        try:
            importlib.import_module(pkg)
            installed_vector_dbs[key] = SupportedVectorDb[key].value
        except ImportError:
            pass

    return enum.Enum("InstalledVectorDb", installed_vector_dbs)


InstalledVectorDb = get_installed_vector_dbs()


class VectorDbMigrateClient(Protocol):
    def migrate(
        self,
        limit: int = DEFAULT_LIMIT,
        batch_size: int | None = None,
    ) -> None:
        ...
