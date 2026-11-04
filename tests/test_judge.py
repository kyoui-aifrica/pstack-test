import json

from builders import cell, grid, p, table

from document_ingestion_pipeline.judge import JevJudge, LLMJudge, RuleBasedJudge


def headings(blocks):
    levels = RuleBasedJudge().judge(blocks)
    return [(levels[i], blocks[i].get("text") or "table") for i in sorted(levels)]


def test_symbols_take_levels_from_where_they_first_appear():
    roman_then_hangul = [
        p("입주 안내서"),
        p("Ⅰ. 개요"),
        p("가. 목적"),
        p("본문"),
        p("Ⅱ. 일정"),
        p("가. 접수"),
        p("본문"),
    ]
    roman_then_arabic = [
        p("입주 안내서"),
        p("Ⅰ. 개요"),
        p("1. 목적"),
        p("본문"),
        p("Ⅱ. 일정"),
        p("1. 접수"),
        p("본문"),
    ]
    assert headings(roman_then_hangul) == [
        (1, "입주 안내서"),
        (2, "Ⅰ. 개요"),
        (3, "가. 목적"),
        (2, "Ⅱ. 일정"),
        (3, "가. 접수"),
    ]
    assert [lv for lv, _ in headings(roman_then_arabic)] == [1, 2, 3, 2, 3]


def test_items_deeper_than_level_three_are_body():
    blocks = [
        p("안내서"),
        p("Ⅰ. 개요"),
        p("1. 목적"),
        p("가. 세부"),
        p("본문"),
    ]
    assert headings(blocks) == [(1, "안내서"), (2, "Ⅰ. 개요"), (3, "1. 목적")]


def test_flat_numbering_is_level_two_and_its_inner_items_are_body():
    blocks = [
        p("분양 공고"),
        p("1. 공급 대상"),
        p("가. 주거 용지"),
        p("본문"),
        p("2. 신청 자격"),
        p("가. 자격 요건"),
        p("본문"),
    ]
    assert headings(blocks) == [
        (1, "분양 공고"),
        (2, "1. 공급 대상"),
        (2, "2. 신청 자격"),
    ]


def test_box_bullets_are_headings_in_a_flat_document():
    blocks = [
        p("재공급 공고문"),
        p("□ 대상 토지 개요"),
        p("ㅇ 위치 : 제주시"),
        p("□ 공급 일정"),
        p("ㅇ 접수 : 12월"),
    ]
    assert headings(blocks) == [
        (1, "재공급 공고문"),
        (2, "□ 대상 토지 개요"),
        (2, "□ 공급 일정"),
    ]


def test_bullets_inside_a_chapter_are_body():
    blocks = [
        p("입주 안내서"),
        p("Ⅰ. 입주안내 개요"),
        p("□ 시설개요"),
        p("ㅇ 위치"),
        p("Ⅱ. 임대료"),
        p("□ 임대보증금"),
        p("본문"),
    ]
    assert headings(blocks) == [
        (1, "입주 안내서"),
        (2, "Ⅰ. 입주안내 개요"),
        (2, "Ⅱ. 임대료"),
    ]


def test_consecutive_numbered_items_without_body_are_a_list():
    blocks = [
        p("분양 공고"),
        p("1. 공급 대상"),
        p("본문"),
        p("2. 제출 서류"),
        p("본문"),
        p("3. 기타 사항"),
        p("본문"),
        p("4. 사업자등록증"),
        p("5. 법인등기부등본"),
        p("6. 인감증명서"),
    ]
    assert headings(blocks) == [
        (1, "분양 공고"),
        (2, "1. 공급 대상"),
        (2, "2. 제출 서류"),
        (2, "3. 기타 사항"),
    ]


def test_numbered_sentences_that_state_facts_are_not_headings():
    blocks = [
        p("분양 공고"),
        p("1. 공급 대상"),
        p("본문"),
        p("2. 신청서는 기한 내에 제출하여야 합니다."),
        p("본문"),
        p("3. 미제출 시 접수 불가함"),
        p("본문"),
    ]
    assert headings(blocks) == [(1, "분양 공고"), (2, "1. 공급 대상")]


def test_table_of_contents_entries_are_not_headings():
    blocks = [
        p("입주 안내서"),
        p("순    서"),
        p("Ⅰ. 개요 ․․․․․․ 1"),
        p("Ⅱ. 일정 ․․․․․․ 2"),
        p("□ 서식 제1호 : 신청서"),
        p("Ⅰ. 개요"),
        p("본문"),
        p("Ⅱ. 일정"),
        p("본문"),
    ]
    assert headings(blocks) == [(1, "입주 안내서"), (2, "Ⅰ. 개요"), (2, "Ⅱ. 일정")]


def test_document_title_ends_at_the_date_and_the_issuer_is_not_a_heading():
    blocks = [
        p("- 스마트빌딩 -"),
        p("입 주 안 내 서"),
        p("2025. 8."),
        p("제주국제자유도시개발센터"),
        p("Ⅰ. 개요"),
        p("본문"),
    ]
    assert headings(blocks) == [
        (1, "- 스마트빌딩 -"),
        (1, "입 주 안 내 서"),
        (2, "Ⅰ. 개요"),
    ]


def test_dates_and_issuer_after_the_body_are_not_headings():
    blocks = [
        p("분양 공고"),
        p("1. 공급 대상"),
        p("본문"),
        p("2026. 03. 10."),
        p("제주국제자유도시개발센터 이사장"),
    ]
    assert headings(blocks) == [(1, "분양 공고"), (2, "1. 공급 대상")]


def test_chapter_title_inside_a_one_row_box_table_is_a_heading():
    blocks = [
        p("입주 안내서"),
        table([[cell("Ⅰ"), cell("입주안내 개요")]]),
        p("본문"),
        table([[cell("Ⅱ"), cell("임대시설 개요")]]),
        p("본문"),
    ]
    levels = RuleBasedJudge().judge(blocks)
    assert levels == {0: 1, 1: 2, 3: 2}


def test_multi_row_tables_are_never_headings():
    blocks = [p("분양 공고"), p("1. 공급 대상"), grid(3, 2), p("본문")]
    assert RuleBasedJudge().judge(blocks) == {0: 1, 1: 2}


def test_forms_under_a_chapter_are_level_three():
    blocks = [
        p("입주 안내서"),
        p("Ⅰ. 개요"),
        p("본문"),
        p("Ⅱ. 첨부"),
        p("□ 서식 제1호 : 신청서"),
        p("[서식 제1호]"),
        grid(3, 3),
        p("[서식 제2호]"),
        p("사업계획서"),
        p("1. 사업개요"),
        grid(3, 3),
    ]
    assert headings(blocks) == [
        (1, "입주 안내서"),
        (2, "Ⅰ. 개요"),
        (2, "Ⅱ. 첨부"),
        (3, "[서식 제1호]"),
        (3, "[서식 제2호]"),
    ]


def test_standalone_forms_in_a_flat_document_are_level_two():
    blocks = [
        p("분양 공고"),
        p("1. 공급 대상"),
        p("본문"),
        p("2. 기타 사항"),
        p("본문"),
        p("[별지 제1호서식]"),
        grid(3, 3),
    ]
    assert headings(blocks) == [
        (1, "분양 공고"),
        (2, "1. 공급 대상"),
        (2, "2. 기타 사항"),
        (2, "[별지 제1호서식]"),
    ]


def test_numbering_that_restarts_is_a_sub_list():
    blocks = [
        p("분양 공고"),
        p("1. 공급 대상"),
        p("본문"),
        p("2. 신청 자격"),
        p("본문"),
        p("1. 개인 신청"),
        p("본문"),
        p("2. 법인 신청"),
        p("본문"),
        p("3. 공급 일정"),
        p("본문"),
    ]
    assert headings(blocks) == [
        (1, "분양 공고"),
        (2, "1. 공급 대상"),
        (2, "2. 신청 자격"),
        (2, "3. 공급 일정"),
    ]


def test_repeated_number_is_still_a_heading():
    blocks = [
        p("분양 공고"),
        p("6. 정산"),
        p("본문"),
        p("6. 소유권 이전"),
        p("본문"),
    ]
    assert [t for _, t in headings(blocks)] == [
        "분양 공고",
        "6. 정산",
        "6. 소유권 이전",
    ]


def test_flat_document_trailing_box_with_heading_font_is_level_two():
    blocks = [
        p("분양공고문", size=27),
        p("① 공급 대상", size=16),
        p("ㅇ 위치", size=14),
        p("② 신청 자격", size=16),
        p("ㅇ 자격", size=14),
        p("□ 접수처 : 첨단운영처", size=16),
        p("ㅇ 전화", size=15),
    ]
    assert headings(blocks) == [
        (1, "분양공고문"),
        (2, "① 공급 대상"),
        (2, "② 신청 자격"),
        (2, "□ 접수처 : 첨단운영처"),
    ]


class FakeCompletions:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        content = json.dumps(self.answers.pop(0), ensure_ascii=False)
        message = type("M", (), {"content": content})
        return type("R", (), {"choices": [type("C", (), {"message": message})]})


def fake_client(answers):
    completions = FakeCompletions(answers)
    chat = type("Chat", (), {"completions": completions})
    return type("Client", (), {"chat": chat})(), completions


def test_llm_judge_keeps_headings_whose_text_matches_the_block():
    blocks = [p("분양 공고"), p("1. 공급 대상"), p("본문")]
    client, completions = fake_client(
        [
            {
                "headings": [
                    {"block": 0, "level": 1, "text": "분양 공고"},
                    {"block": 2, "level": 2, "text": "1. 공급 대상"},
                    {"block": 1, "level": 2, "text": "1.공급  대상"},
                    {"block": 9, "level": 2, "text": "없는 블록"},
                ]
            }
        ]
    )
    levels = LLMJudge(client=client, model="m").judge(blocks)
    assert levels == {0: 1, 1: 2}
    assert len(completions.prompts) == 1
    assert "[1] (p1) 1. 공급 대상" in completions.prompts[0]


def test_jev_judge_drops_low_confidence_answers():
    blocks = [p("분양 공고"), p("1. 공급 대상"), p("2. 신청 자격")]
    client, _ = fake_client(
        [
            {"is_heading": True, "confidence": 0.95},
            {"is_heading": True, "confidence": 0.9},
            {"is_heading": True, "confidence": 0.4},
            {"level": 1, "confidence": 0.9},
            {"level": 2, "confidence": 0.5},
        ]
    )
    levels = JevJudge(client=client, model="m", workers=1).judge(blocks)
    assert levels == {0: 1}
