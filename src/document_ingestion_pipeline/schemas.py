from typing import Any

from .config import ModelConfig
from .parser import TABLE_TAG_PATTERN

CHILDREN_ANALYZER = "children_text"
TABLE_TAG_FILTER = "renderer_table_tags"


def postgres_ddl(config: ModelConfig) -> list[str]:
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE EXTENSION IF NOT EXISTS pg_search",
        """CREATE TABLE IF NOT EXISTS parents (
            id text PRIMARY KEY,
            text text NOT NULL,
            metadata jsonb NOT NULL
        )""",
        f"""CREATE TABLE IF NOT EXISTS children (
            id text PRIMARY KEY,
            parent_id text NOT NULL REFERENCES parents (id) ON DELETE CASCADE,
            text text NOT NULL,
            embedding vector({config.dimension}) NOT NULL,
            metadata jsonb NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS parents_source_idx"
        " ON parents ((metadata ->> 'source'))",
        "CREATE INDEX IF NOT EXISTS children_parent_id_idx ON children (parent_id)",
        "CREATE INDEX IF NOT EXISTS children_embedding_idx"
        " ON children USING hnsw (embedding vector_cosine_ops)",
        # pg_search 0.26 ignores key_field and allows one bm25 index per table.
        "CREATE INDEX IF NOT EXISTS children_bm25_idx"
        " ON children USING bm25 (id, (text::pdb.lindera('korean')), metadata)",
    ]


def opensearch_index_names(config: ModelConfig) -> tuple[str, str]:
    return f"{config.index_name}-parents", f"{config.index_name}-children"


_METADATA_MAPPING = {
    "properties": {
        "source": {"type": "keyword"},
        "title": {"type": "keyword"},
        "pages": {"type": "integer"},
    }
}


def opensearch_parents_body() -> dict[str, Any]:
    return {
        "mappings": {
            "properties": {
                "text": {"type": "text", "index": False},
                "metadata": _METADATA_MAPPING,
            }
        }
    }


def opensearch_children_body(config: ModelConfig) -> dict[str, Any]:
    return {
        "settings": {
            "index": {"knn": True},
            "analysis": {
                "char_filter": {
                    TABLE_TAG_FILTER: {
                        "type": "pattern_replace",
                        "pattern": TABLE_TAG_PATTERN,
                        "replacement": " ",
                        "flags": "CASE_INSENSITIVE",
                    }
                },
                "analyzer": {
                    CHILDREN_ANALYZER: {
                        "type": "custom",
                        "char_filter": [TABLE_TAG_FILTER],
                        "tokenizer": "nori_tokenizer",
                    }
                },
            },
        },
        "mappings": {
            "properties": {
                "parent_id": {"type": "keyword"},
                "text": {"type": "text", "analyzer": CHILDREN_ANALYZER},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": config.dimension,
                    "method": {"name": "hnsw", "space_type": "cosinesimil"},
                },
                "metadata": _METADATA_MAPPING,
            }
        },
    }
