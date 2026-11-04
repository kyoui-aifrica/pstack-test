import pytest

from document_ingestion_pipeline.config import ConfigError, ModelConfig
from document_ingestion_pipeline.embedder import (
    EmbeddingError,
    embed_documents,
    embedding_client,
)

PLAIN = ModelConfig("nlpai-lab/KURE-v1", "kure-v1", 4)
GEMMA = ModelConfig(
    "google/embeddinggemma-2", "embeddinggemma-2", 4, "title: {title} | text: "
)


def test_all_children_go_out_in_one_request_and_come_back_in_input_order(
    fake_embedding_api,
):
    vectors = embed_documents(["가", "나", "다"], PLAIN, fake_embedding_api.client())

    assert len(fake_embedding_api.requests) == 1
    request = fake_embedding_api.requests[0]
    assert request["path"] == "/v1/embeddings"
    assert request["model"] == "nlpai-lab/KURE-v1"
    assert request["input"] == ["가", "나", "다"]
    assert vectors == [[0.0] * 4, [1.0] * 4, [2.0] * 4]


def test_no_texts_sends_no_request(fake_embedding_api):
    assert embed_documents([], PLAIN, fake_embedding_api.client()) == []
    assert fake_embedding_api.requests == []


def test_plain_prefix_is_prepended_to_every_document(fake_embedding_api):
    config = ModelConfig("m", "m", 4, "passage: ")

    embed_documents(["# 제목\n본문", "본문"], config, fake_embedding_api.client())

    assert fake_embedding_api.requests[0]["input"] == [
        "passage: # 제목\n본문",
        "passage: 본문",
    ]


def test_title_prefix_moves_the_document_title_line_into_the_prefix(
    fake_embedding_api,
):
    text = "# 제주첨단과학기술단지 분양 공고\n## 1. 공급 대상\n본문"

    embed_documents([text], GEMMA, fake_embedding_api.client())

    assert fake_embedding_api.requests[0]["input"] == [
        "title: 제주첨단과학기술단지 분양 공고 | text: ## 1. 공급 대상\n본문"
    ]


@pytest.mark.parametrize("text", ["## 1. 공급 대상\n본문", "본문만 있는 청크"])
def test_title_prefix_uses_none_when_chunk_has_no_document_title(
    fake_embedding_api, text
):
    embed_documents([text], GEMMA, fake_embedding_api.client())

    assert fake_embedding_api.requests[0]["input"] == [f"title: none | text: {text}"]


def test_vector_with_wrong_dimension_raises(fake_embedding_api):
    config = ModelConfig("m", "m", 1024)

    with pytest.raises(EmbeddingError, match="1024"):
        embed_documents(["가"], config, fake_embedding_api.client())


def test_client_url_joins_base_url_port_and_v1(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://gpu.local/")
    monkeypatch.setenv("EMBEDDING_PORT", "8080")
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret")

    client = embedding_client()

    assert str(client.base_url) == "http://gpu.local:8080/v1/"
    assert client.api_key == "secret"


def test_client_url_without_port_and_key_still_builds(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embed.example.com")
    monkeypatch.delenv("EMBEDDING_PORT", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    assert str(embedding_client().base_url) == "https://embed.example.com/v1/"


def test_base_url_without_scheme_defaults_to_http(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "10.0.0.5")
    monkeypatch.setenv("EMBEDDING_PORT", "8080")

    assert str(embedding_client().base_url) == "http://10.0.0.5:8080/v1/"


def test_missing_base_url_raises(monkeypatch):
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)

    with pytest.raises(ConfigError, match="EMBEDDING_BASE_URL"):
        embedding_client()
