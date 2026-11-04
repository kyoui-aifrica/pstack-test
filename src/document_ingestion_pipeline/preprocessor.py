import copy
import re

from .judge import Judge
from .parser import Block, Document, block_text, render

_IMAGE_MARK = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def remove_images(blocks: list[Block]) -> list[Block]:
    kept = []
    for block in blocks:
        if block["type"] == "image":
            continue
        if block["type"] == "table":
            for cell in (c for row in block["table"]["cells"] for c in row if c):
                cell["text"] = _IMAGE_MARK.sub("", cell.get("text", "")).strip()
                if "blocks" in cell:
                    cell["blocks"] = remove_images(cell["blocks"])
                    if not cell["blocks"]:
                        del cell["blocks"]
            if not block_text(block) and not _has_cell_blocks(block):
                continue
        kept.append(block)
    return kept


def _has_cell_blocks(table: Block) -> bool:
    return any(c and c.get("blocks") for row in table["table"]["cells"] for c in row)


def _is_frame(block: Block) -> bool:
    return (
        block["type"] == "table"
        and block["table"]["cols"] == 1
        and _has_cell_blocks(block)
    )


def unwrap_frames(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for block in blocks:
        if not _is_frame(block):
            out.append(block)
            continue
        for row in block["table"]["cells"]:
            for cell in (c for c in row if c):
                if cell.get("blocks"):
                    out.extend(unwrap_frames(cell["blocks"]))
                elif text := cell.get("text", "").strip():
                    out.append(
                        {
                            "type": "paragraph",
                            "text": text,
                            "pageNumber": block.get("pageNumber"),
                        }
                    )
    return out


def demote_headings(blocks: list[Block]) -> list[Block]:
    for block in blocks:
        if block["type"] == "heading":
            block["type"] = "paragraph"
            block.pop("level", None)
    return blocks


def apply_levels(blocks: list[Block], levels: dict[int, int]) -> list[Block]:
    for i, level in levels.items():
        block = blocks[i]
        if block["type"] == "table":
            block["text"] = block_text(block)
            del block["table"]
        block["type"] = "heading"
        block["level"] = level
    return blocks


def merge_title(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for block in blocks:
        prev = out[-1] if out else None
        if prev is not None and _is_title(prev) and _is_title(block):
            prev["text"] = f"{prev['text']}\n{block['text']}"
            continue
        out.append(block)
    return out


def _is_title(block: Block) -> bool:
    return block["type"] == "heading" and block.get("level") == 1


def outline(blocks: list[Block]) -> list[dict]:
    return [
        {"level": b["level"], "text": b["text"], "pageNumber": b.get("pageNumber")}
        for b in blocks
        if b["type"] == "heading"
    ]


def preprocess(parsed: Document, judge: Judge) -> Document:
    blocks = copy.deepcopy(parsed["blocks"])
    blocks = demote_headings(unwrap_frames(remove_images(blocks)))
    blocks = merge_title(apply_levels(blocks, judge.judge(blocks)))

    by_page: dict[int | None, list[Block]] = {}
    for block in blocks:
        by_page.setdefault(block.get("pageNumber"), []).append(block)
    markdown, *page_markdown = render([blocks, *by_page.values()])
    return {
        **parsed,
        "blocks": blocks,
        "markdown": markdown,
        "pages": [
            {"pageNumber": number, "markdown": md}
            for number, md in zip(by_page, page_markdown, strict=True)
        ],
        "outline": outline(blocks),
    }
