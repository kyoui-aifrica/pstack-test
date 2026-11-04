import copy

from builders import cell, grid, h, image, p, table

from document_ingestion_pipeline.judge import RuleBasedJudge
from document_ingestion_pipeline.preprocessor import preprocess


class FixedJudge:
    def __init__(self, levels):
        self.levels = levels
        self.seen = None

    def judge(self, blocks):
        self.seen = copy.deepcopy(blocks)
        return self.levels


def doc(blocks):
    return {
        "source": "공고.hwp",
        "blocks": blocks,
        "markdown": "",
        "pages": [],
        "outline": [],
    }


def test_kordoc_headings_are_replaced_by_the_judge_result():
    parsed = doc([h("2026. 03. 10.", 3), p("1. 공급 대상"), p("본문")])
    result = preprocess(parsed, FixedJudge({1: 2}))
    assert [(b["type"], b.get("level")) for b in result["blocks"]] == [
        ("paragraph", None),
        ("heading", 2),
        ("paragraph", None),
    ]
    assert result["outline"] == [{"level": 2, "text": "1. 공급 대상", "pageNumber": 1}]
    assert "## 1. 공급 대상" in result["markdown"]
    assert "#" not in result["markdown"].split("## 1.")[0]


def test_judge_sees_kordoc_headings_as_body():
    judge = FixedJudge({})
    preprocess(doc([h("제주국제자유도시개발센터 이사장", 1)]), judge)
    assert judge.seen == [
        {
            "type": "paragraph",
            "text": "제주국제자유도시개발센터 이사장",
            "pageNumber": 1,
        }
    ]


def test_separately_printed_form_label_becomes_the_form_heading():
    parsed = doc(
        [
            p("입주 안내서"),
            p("Ⅰ. 첨부"),
            p("□ 서식 제1호 : 신청서"),
            h("[서식 제1호]", 2),
            p("지식산업센터 입주자 사업계획서"),
            grid(3, 3),
        ]
    )
    result = preprocess(parsed, RuleBasedJudge())
    assert [(o["level"], o["text"]) for o in result["outline"]] == [
        (1, "입주 안내서"),
        (2, "Ⅰ. 첨부"),
        (3, "[서식 제1호]"),
    ]


def test_multi_line_document_title_is_merged_into_one_heading():
    parsed = doc(
        [
            p("- 스마트빌딩 -"),
            p("입 주 안 내 서"),
            p("2025. 8."),
            p("Ⅰ. 개요"),
            p("본문"),
        ]
    )
    result = preprocess(parsed, RuleBasedJudge())
    assert result["outline"][0] == {
        "level": 1,
        "text": "- 스마트빌딩 -\n입 주 안 내 서",
        "pageNumber": 1,
    }
    assert [o["level"] for o in result["outline"]] == [1, 2]


def test_images_are_removed_from_blocks_and_table_cells():
    parsed = doc(
        [
            p("위치도"),
            image(),
            table([[cell("기정"), cell("![image](image_001.jpg)")]]),
            table([[cell("![image](image_002.png)", blocks=[image()])]]),
        ]
    )
    result = preprocess(parsed, FixedJudge({}))
    assert [b["type"] for b in result["blocks"]] == ["paragraph", "table"]
    assert result["blocks"][1]["table"]["cells"][0][1]["text"] == ""
    assert "image" not in result["markdown"]


def test_one_column_frame_tables_are_unwrapped_into_their_blocks():
    inner = [p("1. 공급 대상"), grid(2, 2), p("본문")]
    frame = table([[cell("", blocks=inner)]], page=2)
    result = preprocess(doc([p("분양 공고"), frame]), FixedJudge({}))
    assert [b["type"] for b in result["blocks"]] == [
        "paragraph",
        "paragraph",
        "table",
        "paragraph",
    ]


def test_heading_table_uses_its_cell_text():
    parsed = doc([p("안내서"), table([[cell("Ⅰ"), cell("입주안내 개요")]]), p("본문")])
    result = preprocess(parsed, RuleBasedJudge())
    assert result["blocks"][1] == {
        "type": "heading",
        "level": 2,
        "text": "Ⅰ 입주안내 개요",
        "pageNumber": 1,
    }


def test_pages_markdown_is_rebuilt_per_page():
    parsed = doc([p("분양 공고"), p("1. 공급 대상", page=2), p("본문", page=2)])
    result = preprocess(parsed, RuleBasedJudge())
    assert [pg["pageNumber"] for pg in result["pages"]] == [1, 2]
    assert result["pages"][1]["markdown"].startswith("## 1. 공급 대상")


def test_parsed_input_is_not_modified():
    parsed = doc([h("제목", 1), image(), p("1. 공급 대상")])
    before = copy.deepcopy(parsed)
    preprocess(parsed, RuleBasedJudge())
    assert parsed == before
