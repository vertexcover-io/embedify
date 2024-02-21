import enum
import importlib
from typing import Protocol


class SupportedVectorDb(enum.Enum):
    Weaviate = "weaviate"


class SupportedEmbedding(enum.Enum):
    OpenAI = "openai"


__vector_db_pkg_map__ = {
    SupportedVectorDb.Weaviate.name: "weaviate",
}


def get_installed_vector_dbs():
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
    def migrate(self, batch_size=None, limit=None):
        pass
