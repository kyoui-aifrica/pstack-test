import json
from dataclasses import dataclass, field

import httpx2
import pytest
from openai import OpenAI


@dataclass
class FakeEmbeddingAPI:
    """OpenAI-compatible /v1/embeddings that returns data in reverse index order."""

    dimension: int = 4
    requests: list[dict] = field(default_factory=list)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        self.requests.append({"path": request.url.path, **payload})
        data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": [float(i)] * self.dimension,
            }
            for i in range(len(payload["input"]))
        ]
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": payload["model"],
                "data": data[::-1],
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            },
        )

    def client(self) -> OpenAI:
        return OpenAI(
            base_url="http://embedding.test/v1",
            api_key="test",
            http_client=httpx2.Client(transport=httpx2.MockTransport(self.handle)),
        )


@pytest.fixture
def fake_embedding_api() -> FakeEmbeddingAPI:
    return FakeEmbeddingAPI()
