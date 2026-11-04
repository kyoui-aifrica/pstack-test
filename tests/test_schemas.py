import re

from document_ingestion_pipeline.config import ModelConfig
from document_ingestion_pipeline.schemas import (
    opensearch_children_body,
    opensearch_parents_body,
    postgres_ddl,
)

CONFIG = ModelConfig("google/embeddinggemma-2", "embeddinggemma-2", 768)


def apply_char_filters(body: dict, field: str, text: str) -> str:
    analysis = body["settings"]["analysis"]
    analyzer = analysis["analyzer"][body["mappings"]["properties"][field]["analyzer"]]
    for name in analyzer["char_filter"]:
        spec = analysis["char_filter"][name]
        assert spec["type"] == "pattern_replace"
        flags = re.IGNORECASE if "CASE_INSENSITIVE" in spec.get("flags", "") else 0
        text = re.sub(spec["pattern"], spec["replacement"], text, flags=flags)
    return text


def test_children_analyzer_strips_renderer_table_tags_but_keeps_angle_bracket_text():
    rendered = (
        '<table><thead><tr><th colspan="2">구분</th></tr></thead>'
        "<tbody><TR><TD>면적<br>(㎡)</TD><td>&lt;공급 대상 토지 위치도&gt;</td></tr>"
        "</tbody></table>\n<공급 대상 토지 위치도> 참조"
    )

    filtered = apply_char_filters(opensearch_children_body(CONFIG), "text", rendered)

    assert re.findall(r"</?\w+[^>]*>", filtered) == ["<공급 대상 토지 위치도>"]
    assert "구분" in filtered and "면적" in filtered and "(㎡)" in filtered


def test_children_text_is_tokenized_with_nori():
    body = opensearch_children_body(CONFIG)
    analyzer_name = body["mappings"]["properties"]["text"]["analyzer"]

    assert body["settings"]["analysis"]["analyzer"][analyzer_name]["tokenizer"] == (
        "nori_tokenizer"
    )


def test_children_embedding_is_cosine_hnsw_with_model_dimension():
    body = opensearch_children_body(CONFIG)

    assert body["settings"]["index"]["knn"] is True
    assert body["mappings"]["properties"]["embedding"] == {
        "type": "knn_vector",
        "dimension": 768,
        "method": {"name": "hnsw", "space_type": "cosinesimil"},
    }


def test_parent_text_is_stored_but_not_indexed():
    text = opensearch_parents_body()["mappings"]["properties"]["text"]

    assert text == {"type": "text", "index": False}


def test_metadata_source_and_title_are_keywords_and_pages_are_integers():
    for body in (opensearch_parents_body(), opensearch_children_body(CONFIG)):
        assert body["mappings"]["properties"]["metadata"]["properties"] == {
            "source": {"type": "keyword"},
            "title": {"type": "keyword"},
            "pages": {"type": "integer"},
        }


def test_postgres_embedding_column_uses_model_dimension():
    ddl = "\n".join(postgres_ddl(CONFIG))

    assert "embedding vector(768)" in ddl
    assert "vector_cosine_ops" in ddl
    assert "pdb.lindera('korean')" in ddl
