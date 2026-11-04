import base64
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import psycopg
import pytest

from document_ingestion_pipeline.config import ModelConfig
from document_ingestion_pipeline.ingester import (
    OpenSearchStore,
    PostgresStore,
    StoreError,
    get_store,
    opensearch_client,
)

KURE = ModelConfig("nlpai-lab/KURE-v1", "kure-v1", 3)
GEMMA = ModelConfig("google/embeddinggemma-2", "embeddinggemma-2", 2)


class FakeIndices:
    def __init__(self, cluster: "FakeOpenSearch") -> None:
        self.cluster = cluster

    def exists(self, index):
        return index in self.cluster.bodies

    def create(self, index, body):
        self.cluster.bodies[index] = body
        self.cluster.documents[index] = {}

    def get_mapping(self, index):
        return {index: {"mappings": self.cluster.bodies[index]["mappings"]}}


class FakeOpenSearch:
    def __init__(self) -> None:
        self.bodies: dict[str, dict] = {}
        self.documents: dict[str, dict[str, dict]] = {}
        self.indices = FakeIndices(self)
        self.deletes: list[str] = []

    def _matching(self, index, body):
        source = body["query"]["term"]["metadata.source"]
        return [
            doc_id
            for doc_id, doc in self.documents[index].items()
            if doc["metadata"]["source"] == source
        ]

    def delete_by_query(self, index, body, **_):
        self.deletes.append(index)
        for doc_id in self._matching(index, body):
            del self.documents[index][doc_id]
        return {}

    def bulk(self, body, **_):
        for action, document in zip(body[::2], body[1::2], strict=True):
            target = action["index"]
            self.documents[target["_index"]][target["_id"]] = document
        return {"errors": False, "items": []}

    def count(self, index, body):
        return {"count": len(self._matching(index, body))}


def parents(source: str, children_per_parent: list[int]) -> list[dict]:
    metadata = {"source": source, "title": "공고", "pages": [1]}
    return [
        {
            "text": f"# 공고\n## 장 {i}",
            "metadata": metadata,
            "children": [
                {"text": f"# 공고\n## 장 {i}\n본문 {j}", "metadata": metadata}
                for j in range(n)
            ],
        }
        for i, n in enumerate(children_per_parent)
    ]


def vectors(count: int, dimension: int) -> list[list[float]]:
    return [[float(i)] * dimension for i in range(count)]


def test_each_model_gets_its_own_parents_and_children_indices():
    cluster = FakeOpenSearch()

    OpenSearchStore(KURE, cluster).ingest("a.hwp", parents("a.hwp", [1]), vectors(1, 3))
    OpenSearchStore(GEMMA, cluster).ingest(
        "a.hwp", parents("a.hwp", [1]), vectors(1, 2)
    )

    assert set(cluster.bodies) == {
        "kure-v1-parents",
        "kure-v1-children",
        "embeddinggemma-2-parents",
        "embeddinggemma-2-children",
    }
    dimensions = {
        index: body["mappings"]["properties"]["embedding"]["dimension"]
        for index, body in cluster.bodies.items()
        if index.endswith("-children")
    }
    assert dimensions == {"kure-v1-children": 3, "embeddinggemma-2-children": 2}


def test_reingesting_a_document_keeps_record_counts_identical():
    store = OpenSearchStore(KURE, FakeOpenSearch())

    store.ingest("a.hwp", parents("a.hwp", [2, 1]), vectors(3, 3))
    first = store.count("a.hwp")
    store.ingest("a.hwp", parents("a.hwp", [2, 1]), vectors(3, 3))

    assert first == (2, 3)
    assert store.count("a.hwp") == first


def test_reingest_replaces_only_that_documents_records():
    store = OpenSearchStore(KURE, FakeOpenSearch())
    store.ingest("a.hwp", parents("a.hwp", [3, 3]), vectors(6, 3))
    store.ingest("b.hwp", parents("b.hwp", [1]), vectors(1, 3))

    store.ingest("a.hwp", parents("a.hwp", [1]), vectors(1, 3))

    assert store.count("a.hwp") == (1, 1)
    assert store.count("b.hwp") == (1, 1)


def test_children_carry_their_embedding_and_parent_id():
    cluster = FakeOpenSearch()

    OpenSearchStore(KURE, cluster).ingest(
        "a.hwp", parents("a.hwp", [1, 1]), [[1.0, 0, 0], [0, 1.0, 0]]
    )

    children = cluster.documents["kure-v1-children"]
    parent_ids = set(cluster.documents["kure-v1-parents"])
    assert len(children) == 2
    assert {doc["parent_id"] for doc in children.values()} == parent_ids
    assert sorted(doc["embedding"] for doc in children.values()) == [
        [0, 1.0, 0],
        [1.0, 0, 0],
    ]


def test_dimension_mismatch_stops_before_deleting_existing_data():
    cluster = FakeOpenSearch()
    OpenSearchStore(ModelConfig("old", "kure-v1", 2), cluster).ingest(
        "a.hwp", parents("a.hwp", [1]), vectors(1, 2)
    )
    cluster.deletes.clear()

    with pytest.raises(StoreError, match="2"):
        OpenSearchStore(KURE, cluster).ingest(
            "a.hwp", parents("a.hwp", [1]), vectors(1, 3)
        )

    assert cluster.deletes == []
    assert OpenSearchStore(KURE, cluster).count("a.hwp") == (1, 1)


def test_embeddings_not_matching_children_count_are_rejected_before_writing():
    cluster = FakeOpenSearch()

    with pytest.raises(ValueError):
        OpenSearchStore(KURE, cluster).ingest(
            "a.hwp", parents("a.hwp", [2]), vectors(1, 3)
        )

    assert cluster.bodies == {}


def test_store_is_chosen_by_name(monkeypatch):
    monkeypatch.setenv("OPENSEARCH_URL", "http://localhost")

    assert isinstance(get_store("postgres", KURE), PostgresStore)
    assert isinstance(get_store("opensearch", KURE), OpenSearchStore)


@pytest.fixture
def opensearch_http(monkeypatch):
    seen: list[dict[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append({k.lower(): v for k, v in self.headers.items()})
            body = json.dumps({"version": {"number": "3.9.0"}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENSEARCH_URL", "http://127.0.0.1")
    monkeypatch.setenv("OPENSEARCH_PORT", str(server.server_port))
    for name in ("OPENSEARCH_USER", "OPENSEARCH_PASSWORD", "OPENSEARCH_CERT_FILE"):
        monkeypatch.delenv(name, raising=False)
    yield seen
    server.shutdown()


def test_opensearch_connects_to_url_and_port_with_basic_auth(
    opensearch_http, monkeypatch
):
    monkeypatch.setenv("OPENSEARCH_USER", "admin")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "pa:ss")

    opensearch_client().info()

    expected = "Basic " + base64.b64encode(b"admin:pa:ss").decode()
    assert [request["authorization"] for request in opensearch_http] == [expected]


def test_opensearch_without_user_sends_no_credentials(opensearch_http):
    opensearch_client().info()

    assert len(opensearch_http) == 1
    assert "authorization" not in opensearch_http[0]


def test_opensearch_cert_file_turns_on_verified_tls(monkeypatch, tmp_path):
    cert = tmp_path / "root-ca.pem"
    cert.write_text("unused")
    monkeypatch.setenv("OPENSEARCH_URL", "https://search.internal")
    monkeypatch.setenv("OPENSEARCH_PORT", "9200")
    monkeypatch.setenv("OPENSEARCH_CERT_FILE", str(cert))

    pool = opensearch_client().transport.get_connection().pool

    assert (pool.scheme, pool.host, pool.port) == ("https", "search.internal", 9200)
    assert pool.ca_certs == str(cert)
    assert pool.cert_reqs == "CERT_REQUIRED"


def test_postgres_connects_with_standard_pg_variables(monkeypatch):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    monkeypatch.setenv("PGHOST", "127.0.0.1")
    monkeypatch.setenv("PGPORT", str(closed_port))
    monkeypatch.setenv("PGCONNECT_TIMEOUT", "2")

    with pytest.raises(psycopg.OperationalError, match=f"port {closed_port}"):
        PostgresStore(KURE).count("a.hwp")
