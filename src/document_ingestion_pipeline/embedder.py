import os

from openai import OpenAI

from .config import ConfigError, ModelConfig

TITLE_PLACEHOLDER = "{title}"
NO_TITLE = "none"


class EmbeddingError(Exception):
    pass


def embedding_client() -> OpenAI:
    base_url = os.environ.get("EMBEDDING_BASE_URL", "").rstrip("/")
    if not base_url:
        raise ConfigError("EMBEDDING_BASE_URL이 설정되지 않았습니다")
    if port := os.environ.get("EMBEDDING_PORT"):
        base_url = f"{base_url}:{port}"
    api_key = os.environ.get("EMBEDDING_API_KEY") or "unused"
    return OpenAI(base_url=f"{base_url}/v1", api_key=api_key)


def format_document(text: str, prefix: str) -> str:
    if TITLE_PLACEHOLDER not in prefix:
        return prefix + text
    first_line, _, rest = text.partition("\n")
    if first_line.startswith("# "):
        title, body = first_line[2:].strip(), rest.lstrip("\n")
    else:
        title, body = NO_TITLE, text
    return prefix.replace(TITLE_PLACEHOLDER, title) + body


def embed_documents(
    texts: list[str], config: ModelConfig, client: OpenAI
) -> list[list[float]]:
    if not texts:
        return []
    response = client.embeddings.create(
        model=config.name,
        input=[format_document(text, config.document_prefix) for text in texts],
    )
    vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
    if len(vectors) != len(texts):
        raise EmbeddingError(
            f"임베딩 {len(texts)}개를 요청했지만 {len(vectors)}개를 받았습니다"
        )
    for vector in vectors:
        if len(vector) != config.dimension:
            raise EmbeddingError(
                f"{config.name} 임베딩 차원이 {len(vector)}입니다 "
                f"(설정: {config.dimension})"
            )
    return vectors
