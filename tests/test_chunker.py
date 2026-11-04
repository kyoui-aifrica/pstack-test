from builders import cell, grid, h, p, table

from document_ingestion_pipeline import chunker
from document_ingestion_pipeline.chunker import chunk


def doc(blocks):
    return {"source": "공고.hwp", "blocks": blocks}


CHAPTERED = doc(
    [
        h("입주 안내서", 1),
        p("2025. 8."),
        h("Ⅰ. 개요", 2, page=2),
        p("본문 하나", page=2),
        h("Ⅱ. 첨부", 2, page=3),
        p("첨부 안내", page=3),
        h("[서식 제1호]", 3, page=4),
        p("서식 본문", page=4),
    ]
)


def test_every_child_starts_with_its_heading_path():
    children = [c["text"] for parent in chunk(CHAPTERED) for c in parent["children"]]
    assert children[0].startswith("# 입주 안내서")
    assert children[1].startswith("# 입주 안내서\n\n\n## Ⅰ. 개요")
    form = children[-1]
    assert form.startswith("# 입주 안내서\n\n\n## Ⅱ. 첨부\n\n\n### [서식 제1호]")
    assert form.endswith("서식 본문")


def test_each_chapter_is_one_parent_holding_its_sections():
    parents = chunk(CHAPTERED)
    assert [p_["text"].splitlines()[0] for p_ in parents] == ["# 입주 안내서"] * 3
    assert "## Ⅱ. 첨부" in parents[2]["text"]
    assert "### [서식 제1호]" in parents[2]["text"]
    assert len(parents[2]["children"]) == 2


def test_metadata_carries_source_title_and_pages():
    parents = chunk(CHAPTERED)
    assert parents[1]["metadata"] == {
        "source": "공고.hwp",
        "title": "입주 안내서",
        "pages": [2],
    }
    assert parents[2]["metadata"]["pages"] == [3, 4]
    assert [c["metadata"]["pages"] for c in parents[2]["children"]] == [[3], [4]]


def test_table_is_its_own_child_with_the_sentence_before_it():
    parents = chunk(
        doc(
            [
                h("공고", 1),
                h("1. 공급 대상", 2),
                p("앞 본문"),
                p("공급 대상 토지는 아래와 같다."),
                grid(2, 2),
                p("뒤 본문"),
            ]
        )
    )
    texts = [c["text"] for c in parents[0]["children"]]
    assert len(texts) == 3
    assert texts[0].endswith("앞 본문")
    assert "공급 대상 토지는 아래와 같다.\n\n<table>" in texts[1]
    assert texts[2].endswith("뒤 본문")
    assert sum("공급 대상 토지는 아래와 같다." in t for t in texts) == 1


def test_one_row_box_tables_are_treated_as_text():
    parents = chunk(
        doc(
            [
                h("공고", 1),
                h("1. 유의사항", 2),
                p("앞 본문"),
                table([[cell("변경사항은 홈페이지에 게시함")]]),
                p("뒤 본문"),
            ]
        )
    )
    [child] = parents[0]["children"]
    assert "<table" not in child["text"]
    assert "앞 본문" in child["text"] and "뒤 본문" in child["text"]


def test_long_sections_are_split_under_the_size_limit(monkeypatch):
    monkeypatch.setattr(chunker, "MAX_CHILD_CHARS", 30)
    parents = chunk(
        doc(
            [h("공고", 1), h("1. 기타", 2)]
            + [p(f"{i}번째 문장입니다 열두자") for i in range(6)]
        )
    )
    children = parents[0]["children"]
    assert len(children) == 3
    assert all(c["text"].startswith("# 공고\n\n\n## 1. 기타") for c in children)


def test_document_without_blocks_has_no_chunks():
    assert chunk(doc([])) == []
