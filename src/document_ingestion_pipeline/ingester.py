from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Protocol

import psycopg
from opensearchpy import OpenSearch
from psycopg.types.json import Jsonb

from .config import ConfigError, ModelConfig
from .schemas import (
    opensearch_children_body,
    opensearch_index_names,
    opensearch_parents_body,
    postgres_ddl,
)

if TYPE_CHECKING:
    from .chunker import Child, Parent

Record = dict[str, Any]


class StoreError(Exception):
    pass


class Store(Protocol):
    def ingest(
        self, source: str, parents: list[Parent], embeddings: list[list[float]]
    ) -> None: ...

    def count(self, source: str) -> tuple[int, int]: ...


def _records(
    source: str, parents: list[Parent], embeddings: list[list[float]]
) -> tuple[list[Record], list[Record]]:
    parent_records: list[Record] = []
    children: list[tuple[str, str, Child]] = []
    for i, parent in enumerate(parents):
        parent_id = f"{source}:{i}"
        parent_records.append(
            {"id": parent_id, "text": parent["text"], "metadata": parent["metadata"]}
        )
        children += [
            (f"{parent_id}:{j}", parent_id, child)
            for j, child in enumerate(parent["children"])
        ]
    child_records = [
        {
            "id": child_id,
            "parent_id": parent_id,
            "text": child["text"],
            "embedding": vector,
            "metadata": child["metadata"],
        }
        for (child_id, parent_id, child), vector in zip(
            children, embeddings, strict=True
        )
    ]
    return parent_records, child_records


class PostgresStore:
    """Connects with libpq defaults, so the standard PG* variables apply."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def ingest(
        self, source: str, parents: list[Parent], embeddings: list[list[float]]
    ) -> None:
        parent_records, child_records = _records(source, parents, embeddings)
        with psycopg.connect("") as conn, conn.cursor() as cur:
            for statement in postgres_ddl(self.config):
                cur.execute(statement)
            cur.execute(
                "DELETE FROM parents WHERE metadata ->> 'source' = %s", (source,)
            )
            cur.executemany(
                "INSERT INTO parents (id, text, metadata) VALUES (%s, %s, %s)",
                [(r["id"], r["text"], Jsonb(r["metadata"])) for r in parent_records],
            )
            cur.executemany(
                "INSERT INTO children (id, parent_id, text, embedding, metadata)"
                " VALUES (%s, %s, %s, %s::vector, %s)",
                [
                    (
                        r["id"],
                        r["parent_id"],
                        r["text"],
                        json.dumps(r["embedding"]),
                        Jsonb(r["metadata"]),
                    )
                    for r in child_records
                ],
            )

    def count(self, source: str) -> tuple[int, int]:
        with psycopg.connect("") as conn:
            row = conn.execute(
                "SELECT"
                " (SELECT count(*) FROM parents WHERE metadata ->> 'source' = %(s)s),"
                " (SELECT count(*) FROM children WHERE metadata ->> 'source' = %(s)s)",
                {"s": source},
            ).fetchone()
        return row[0], row[1]


def opensearch_client() -> OpenSearch:
    url = os.environ.get("OPENSEARCH_URL", "").rstrip("/")
    if not url:
        raise ConfigError("OPENSEARCH_URL이 설정되지 않았습니다")
    if port := os.environ.get("OPENSEARCH_PORT"):
        url = f"{url}:{port}"
    options: dict[str, Any] = {}
    if user := os.environ.get("OPENSEARCH_USER"):
        options["http_auth"] = (user, os.environ.get("OPENSEARCH_PASSWORD", ""))
    if cert_file := os.environ.get("OPENSEARCH_CERT_FILE"):
        options.update(ca_certs=cert_file, verify_certs=True)
    return OpenSearch(hosts=[url], **options)


def _source_query(source: str) -> dict[str, Any]:
    return {"query": {"term": {"metadata.source": source}}}


class OpenSearchStore:
    def __init__(self, config: ModelConfig, client: OpenSearch | None = None) -> None:
        self.config = config
        self.client = client or opensearch_client()
        self.parents_index, self.children_index = opensearch_index_names(config)

    def _ensure_indices(self) -> None:
        if self.client.indices.exists(index=self.children_index):
            mapping = self.client.indices.get_mapping(index=self.children_index)
            properties = mapping[self.children_index]["mappings"]["properties"]
            dimension = properties["embedding"]["dimension"]
            if dimension != self.config.dimension:
                raise StoreError(
                    f"{self.children_index}의 임베딩 차원은 {dimension}이고 "
                    f"{self.config.name} 설정은 {self.config.dimension}입니다. "
                    "기존 데이터는 그대로 둡니다"
                )
        else:
            self.client.indices.create(
                index=self.children_index, body=opensearch_children_body(self.config)
            )
        if not self.client.indices.exists(index=self.parents_index):
            self.client.indices.create(
                index=self.parents_index, body=opensearch_parents_body()
            )

    def ingest(
        self, source: str, parents: list[Parent], embeddings: list[list[float]]
    ) -> None:
        parent_records, child_records = _records(source, parents, embeddings)
        self._ensure_indices()
        for index in (self.children_index, self.parents_index):
            self.client.delete_by_query(
                index=index, body=_source_query(source), refresh=True
            )
        actions: list[dict[str, Any]] = []
        for index, batch in (
            (self.parents_index, parent_records),
            (self.children_index, child_records),
        ):
            for record in batch:
                document = {k: v for k, v in record.items() if k != "id"}
                actions += [{"index": {"_index": index, "_id": record["id"]}}, document]
        if not actions:
            return
        response = self.client.bulk(body=actions, refresh=True)
        if response["errors"]:
            failed = [
                item["index"]["error"]
                for item in response["items"]
                if "error" in item["index"]
            ]
            raise StoreError(f"OpenSearch 적재 실패 {len(failed)}건: {failed[0]}")

    def count(self, source: str) -> tuple[int, int]:
        query = _source_query(source)
        parents, children = (
            self.client.count(index=index, body=query)["count"]
            for index in (self.parents_index, self.children_index)
        )
        return parents, children


STORES: dict[str, type[PostgresStore] | type[OpenSearchStore]] = {
    "postgres": PostgresStore,
    "opensearch": OpenSearchStore,
}


def get_store(name: str, config: ModelConfig) -> Store:
    return STORES[name](config)
