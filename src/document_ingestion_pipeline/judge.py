import json
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol

from openai import OpenAI

from .parser import Block, block_text, normalize_text

Levels = dict[int, int]

LEVEL_GUIDE = """\
제목 레벨 기준:
- 레벨 1: 문서 제목. 여러 줄에 걸쳐 있으면 모두 레벨 1.
- 레벨 2: 장 제목, 독립된 첨부·서식 제목.
- 레벨 3: 장으로 나뉜 문서(Ⅰ., 제1장 등)의 절 제목, 장 아래의 첨부·서식 제목.
제목이 아닌 것:
- 목차
- 번호가 붙었더라도 사실이나 지시를 서술하는 문장
- 본문 없이 이어지는 나열 항목
- 날짜와 발행 기관명
- 레벨 3보다 깊은 항목
- 장 안의 □, ㅇ 같은 글머리 항목"""


class Judge(Protocol):
    def judge(self, blocks: list[Block]) -> Levels: ...


class Tier(Enum):
    CHAPTER = auto()
    NUMBER = auto()
    BOX = auto()
    BULLET = auto()
    FORM = auto()


@dataclass(frozen=True)
class Marker:
    kind: str
    tier: Tier
    number: int | None = None


_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
_ROMAN = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ"
_HANGUL = "가나다라마바사아자차카타파하"
_ROMAN_ASCII = {"I": 1, "V": 5, "X": 10}


def _roman_ascii(token: str) -> int:
    total = 0
    for i, ch in enumerate(token):
        value = _ROMAN_ASCII[ch]
        nxt = _ROMAN_ASCII[token[i + 1]] if i + 1 < len(token) else 0
        total += -value if value < nxt else value
    return total


_MARKERS: list[tuple[str, Tier, re.Pattern[str], Callable[[str], int] | None]] = [
    (
        "form",
        Tier.FORM,
        re.compile(
            r"^[\[〔【<(]?\s*(?:서식|별지|별첨|붙임|첨부)\s*(?:제\s*)?(\d+)\s*호?"
            r"(?:\s*서식)?\s*[\]〕】>)]"
        ),
        int,
    ),
    ("part", Tier.CHAPTER, re.compile(r"^제\s*(\d+)\s*편(?:\s|$)"), int),
    ("chapter", Tier.CHAPTER, re.compile(r"^제\s*(\d+)\s*장(?:\s|$)"), int),
    (
        "roman",
        Tier.CHAPTER,
        re.compile(rf"^([{_ROMAN}])\s*[.．]?(?:\s|$)"),
        lambda m: _ROMAN.index(m) + 1,
    ),
    ("roman", Tier.CHAPTER, re.compile(r"^([IVX]+)\s*[.．](?:\s|$)"), _roman_ascii),
    ("section", Tier.NUMBER, re.compile(r"^제\s*(\d+)\s*절(?:\s|$)"), int),
    ("arabic", Tier.NUMBER, re.compile(r"^(\d{1,2})\s*[.．](?!\d)"), int),
    ("arabic-paren", Tier.NUMBER, re.compile(r"^(\d{1,2})\s*\)"), int),
    ("paren-arabic", Tier.NUMBER, re.compile(r"^\(\s*(\d{1,2})\s*\)"), int),
    (
        "circled",
        Tier.NUMBER,
        re.compile(rf"^([{_CIRCLED}])"),
        lambda m: _CIRCLED.index(m) + 1,
    ),
    (
        "hangul",
        Tier.NUMBER,
        re.compile(rf"^([{_HANGUL}])\s*[.．](?:\s|$)"),
        lambda m: _HANGUL.index(m) + 1,
    ),
    (
        "hangul-paren",
        Tier.NUMBER,
        re.compile(rf"^([{_HANGUL}])\s*\)"),
        lambda m: _HANGUL.index(m) + 1,
    ),
    ("box", Tier.BOX, re.compile(r"^[□■◆◇▣]"), None),
    ("bullet", Tier.BULLET, re.compile(r"^[ㅇ○◦●•\-–·‧ㆍ※*▶▷►▸☞→]"), None),
]

_DECORATED = re.compile(r"^[-–—]\s*\S.*\S\s*[-–—]$")
_DATE = re.compile(r"^[‘'`’]?\d{2,4}\s*[.．년]\s*\d{1,2}\s*[.．월]")
_LEADER = re.compile(r"(?:[.․·…ㆍ‥]\s*){3,}\s*\d+\s*$")
_TOC_HEADER = re.compile(r"^(?:목차|차례|순서|contents)$", re.IGNORECASE)
_DESCRIPTIVE = re.compile(r"(?:다|함|됨|음)\s*[.。]?\s*$")

MAX_HEADING_LENGTH = 50
MAX_TITLE_LINES = 4


def marker_of(text: str) -> Marker | None:
    text = text.strip()
    if _DECORATED.match(text):
        return None
    for kind, tier, pattern, number in _MARKERS:
        if m := pattern.match(text):
            value = number(m.group(1)) if number and m.groups() else None
            return Marker(kind, tier, value)
    return None


def _is_candidate_block(block: Block) -> bool:
    if block["type"] == "table":
        return block["table"]["rows"] == 1
    return block["type"] in ("paragraph", "heading", "list")


def _font_size(block: Block) -> float | None:
    return (block.get("style") or {}).get("fontSize")


def _looks_like_heading_text(text: str) -> bool:
    return (
        0 < len(text) <= MAX_HEADING_LENGTH
        and not _DESCRIPTIVE.search(text)
        and not _LEADER.search(text)
        and not _DATE.match(text)
    )


def _title_lines(blocks: list[Block]) -> list[int]:
    lines: list[int] = []
    for i, block in enumerate(blocks):
        text = block_text(block).strip()
        if not text:
            continue
        if (
            not _is_candidate_block(block)
            or marker_of(text)
            or _DATE.match(text)
            or len(text) > MAX_HEADING_LENGTH
            or _TOC_HEADER.match(normalize_text(text))
            or len(lines) == MAX_TITLE_LINES
        ):
            break
        lines.append(i)
    return lines


def _toc_lines(blocks: list[Block]) -> set[int]:
    texts = [block_text(b).strip() for b in blocks]
    toc = {i for i, t in enumerate(texts) if _LEADER.search(t)}
    header = next(
        (i for i, t in enumerate(texts) if _TOC_HEADER.match(normalize_text(t))), None
    )
    if header is None or not toc:
        return toc
    entries = [normalize_text(_LEADER.sub("", texts[i])) for i in sorted(toc)]
    first_entry = entries[0]
    end = max(toc)
    for i in range(end + 1, len(texts)):
        if normalize_text(texts[i]) == first_entry:
            end = i - 1
            break
    else:
        end = max(toc)
    return toc | set(range(header, end + 1))


class RuleBasedJudge:
    """Numbering symbols decide levels; each symbol binds to a level per document."""

    def judge(self, blocks: list[Block]) -> Levels:
        levels: Levels = {}
        title = _title_lines(blocks)
        for i in title:
            levels[i] = 1
        skip = set(title) | _toc_lines(blocks)

        candidates: dict[int, Marker] = {}
        for i, block in enumerate(blocks):
            if i in skip or not _is_candidate_block(block):
                continue
            text = block_text(block).strip()
            marker = marker_of(text)
            if (
                marker
                and marker.tier is not Tier.BULLET
                and _looks_like_heading_text(text)
            ):
                candidates[i] = marker

        chaptered = any(m.tier is Tier.CHAPTER for m in candidates.values())
        max_level = 3 if chaptered else 2

        kind_level: dict[str, int] = {}
        last_number: dict[str, int] = {}
        sub_list: set[str] = set()
        top_font: float | None = None
        current = 1
        chapter_seen = False

        for i, marker in sorted(candidates.items()):
            if self._enumerated(blocks, candidates, i):
                continue
            if marker.tier is Tier.BOX and chaptered:
                continue
            if marker.tier is Tier.FORM:
                level = 3 if chapter_seen else 2
            elif marker.kind in kind_level:
                level = kind_level[marker.kind]
                previous = last_number.get(marker.kind)
                if (
                    marker.number is not None
                    and previous is not None
                    and (
                        marker.number < previous
                        or (marker.kind in sub_list and marker.number == previous)
                    )
                ):
                    sub_list.add(marker.kind)
                    continue
                sub_list.discard(marker.kind)
            else:
                level = current + 1
                font = _font_size(blocks[i])
                if (
                    level > max_level
                    and not chaptered
                    and top_font is not None
                    and font is not None
                    and font >= top_font
                ):
                    level = 2
                kind_level[marker.kind] = level
            if level > max_level:
                continue

            levels[i] = level
            current = level
            if level == 2 and top_font is None:
                top_font = _font_size(blocks[i])
            if marker.tier is Tier.CHAPTER:
                chapter_seen = True
            if marker.number is not None and marker.tier is not Tier.FORM:
                last_number[marker.kind] = marker.number
            for kind, kind_lv in kind_level.items():
                if kind_lv > level:
                    last_number.pop(kind, None)
                    sub_list.discard(kind)
        return levels

    @staticmethod
    def _enumerated(blocks: list[Block], candidates: dict[int, Marker], i: int) -> bool:
        kind = candidates[i].kind
        return any(
            0 <= j < len(blocks) and j in candidates and candidates[j].kind == kind
            for j in (i - 1, i + 1)
        )


def _numbered_document(blocks: list[Block]) -> str:
    lines = []
    for i, block in enumerate(blocks):
        text = block_text(block).replace("\n", " ").strip()
        if block["type"] == "table":
            table = block["table"]
            text = f"[표 {table['rows']}x{table['cols']}] {text[:200]}"
        lines.append(f"[{i}] (p{block.get('pageNumber', '?')}) {text}")
    return "\n".join(lines)


def _matches_block(answer: str, block: Block) -> bool:
    expected = normalize_text(block_text(block))
    got = normalize_text(answer)
    return bool(got) and (expected == got or expected.startswith(got))


_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "headings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "block": {"type": "integer"},
                    "level": {"type": "integer", "enum": [1, 2, 3]},
                    "text": {"type": "string"},
                },
                "required": ["block", "level", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["headings"],
    "additionalProperties": False,
}


def _structured(
    client: OpenAI, model: str, prompt: str, schema: dict[str, Any], name: str
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": True},
        },
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content[content.find("{") : content.rfind("}") + 1])


class LLMJudge:
    def __init__(self, client: OpenAI | None = None, model: str | None = None) -> None:
        self.client = client or OpenAI(
            base_url=os.environ["LLM_BASE_URL"], api_key=os.environ.get("LLM_API_KEY")
        )
        self.model = model or os.environ["LLM_MODEL"]

    def judge(self, blocks: list[Block]) -> Levels:
        prompt = (
            "아래는 한국어 공문서를 블록 단위로 나눠 번호를 붙인 것이다. "
            "목차와 번호 체계를 근거로 제목 블록을 모두 찾아 블록 번호, 레벨, "
            "블록 텍스트 그대로를 답하라.\n\n"
            f"{LEVEL_GUIDE}\n\n문서:\n{_numbered_document(blocks)}"
        )
        answer = _structured(self.client, self.model, prompt, _LLM_SCHEMA, "headings")
        levels: Levels = {}
        for item in answer.get("headings", []):
            i, level = item.get("block"), item.get("level")
            if (
                isinstance(i, int)
                and 0 <= i < len(blocks)
                and level in (1, 2, 3)
                and _matches_block(str(item.get("text", "")), blocks[i])
            ):
                levels[i] = level
        return levels


_IS_HEADING_SCHEMA = {
    "type": "object",
    "properties": {
        "is_heading": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["is_heading", "confidence"],
    "additionalProperties": False,
}
_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "level": {"type": "integer", "enum": [1, 2, 3]},
        "confidence": {"type": "number"},
    },
    "required": ["level", "confidence"],
    "additionalProperties": False,
}


class JevJudge:
    """Per-block questions to TypeSafe Jev; low-confidence answers are dropped."""

    MIN_CONFIDENCE = 0.7
    MAX_BLOCK_LENGTH = 100

    def __init__(
        self, client: OpenAI | None = None, model: str | None = None, workers: int = 8
    ) -> None:
        base = os.environ.get("TYPESAFE_BASE_URL", "https://openrouter.ai/api")
        self.client = client or OpenAI(
            base_url=base.rstrip("/") + "/v1",
            api_key=os.environ.get("TYPESAFE_API_KEY"),
        )
        self.model = model or os.environ.get("TYPESAFE_MODEL", "typesafe/jev-router")
        self.workers = workers

    def judge(self, blocks: list[Block]) -> Levels:
        asked = [
            i
            for i, b in enumerate(blocks)
            if _is_candidate_block(b)
            and 0 < len(block_text(b).strip()) <= self.MAX_BLOCK_LENGTH
        ]
        with ThreadPoolExecutor(self.workers) as pool:
            heading = dict(
                zip(
                    asked,
                    pool.map(lambda i: self._is_heading(blocks, i), asked),
                    strict=True,
                )
            )
        levels: Levels = {}
        for i in asked:
            if heading[i]:
                level = self._level(blocks, i, levels)
                if level is not None:
                    levels[i] = level
        return levels

    def _state(self, blocks: list[Block], i: int, levels: Levels | None) -> str:
        page = blocks[i].get("pageNumber")
        same_page = [
            f"[{j}] {block_text(b).replace(chr(10), ' ')[:150]}"
            for j, b in enumerate(blocks)
            if b.get("pageNumber") == page
        ]
        state = (
            f"{LEVEL_GUIDE}\n\n같은 페이지 블록:\n"
            + "\n".join(same_page)
            + f"\n\n판정할 블록 [{i}]: {block_text(blocks[i])}"
        )
        if levels is not None:
            decided = [
                f"레벨 {lv}: {block_text(blocks[j])}"
                for j, lv in sorted(levels.items())
            ]
            state += "\n\n앞서 판정한 제목:\n" + ("\n".join(decided) or "(없음)")
        return state

    def _is_heading(self, blocks: list[Block], i: int) -> bool:
        answer = _structured(
            self.client,
            self.model,
            self._state(blocks, i, None)
            + "\n\n이 블록이 제목인가? confidence는 0~1 확신도.",
            _IS_HEADING_SCHEMA,
            "is_heading",
        )
        return bool(answer.get("is_heading")) and (
            float(answer.get("confidence", 0)) >= self.MIN_CONFIDENCE
        )

    def _level(self, blocks: list[Block], i: int, levels: Levels) -> int | None:
        answer = _structured(
            self.client,
            self.model,
            self._state(blocks, i, levels)
            + "\n\n이 제목의 레벨은? confidence는 0~1 확신도.",
            _LEVEL_SCHEMA,
            "level",
        )
        level = answer.get("level")
        if level in (1, 2, 3) and float(answer.get("confidence", 0)) >= (
            self.MIN_CONFIDENCE
        ):
            return int(level)
        return None


JUDGES: dict[str, Callable[[], Judge]] = {
    "rule-based": RuleBasedJudge,
    "llm": LLMJudge,
    "jev": JevJudge,
}
JUDGE_NAMES = tuple(JUDGES)


def get_judge(name: str) -> Judge:
    if name not in JUDGES:
        raise ValueError(f"알 수 없는 판정기: {name} (가능: {', '.join(JUDGE_NAMES)})")
    return JUDGES[name]()
