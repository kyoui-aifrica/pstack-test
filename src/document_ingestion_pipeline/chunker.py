from dataclasses import dataclass, field
from typing import TypedDict

from .parser import Block, Document, render

MAX_CHILD_CHARS = 10_000


class Metadata(TypedDict):
    source: str
    title: str
    pages: list[int]


class Child(TypedDict):
    text: str
    metadata: Metadata


class Parent(TypedDict):
    text: str
    metadata: Metadata
    children: list[Child]


@dataclass
class _Section:
    path: list[Block]
    blocks: list[Block] = field(default_factory=list)


@dataclass
class _Chapter:
    path: list[Block]
    blocks: list[Block] = field(default_factory=list)
    sections: list[_Section] = field(default_factory=list)


def _is_heading(block: Block, *levels: int) -> bool:
    return block["type"] == "heading" and block.get("level") in levels


def _one_line(heading: Block) -> Block:
    return {**heading, "text": " ".join(heading["text"].split())}


def _chapters(blocks: list[Block]) -> list[_Chapter]:
    title = next((b for b in blocks if _is_heading(b, 1)), None)
    root = [_one_line(title)] if title else []
    chapters = [_Chapter(path=root)]
    for block in blocks:
        if _is_heading(block, 1, 2):
            if chapters[-1].blocks:
                chapters.append(_Chapter(path=root))
            if _is_heading(block, 2):
                chapters[-1].path = [*root, _one_line(block)]
            chapters[-1].blocks.append(block)
            continue
        chapter = chapters[-1]
        chapter.blocks.append(block)
        if _is_heading(block, 3):
            chapter.sections.append(_Section(path=[*chapter.path, _one_line(block)]))
        elif not chapter.sections:
            chapter.sections.append(_Section(path=chapter.path, blocks=[block]))
        else:
            chapter.sections[-1].blocks.append(block)
    return [c for c in chapters if c.blocks]


def _child_groups(section: _Section, is_table: list[bool]) -> list[list[Block]]:
    """Text runs and tables; each table takes the sentence just before it."""
    groups: list[list[Block]] = []
    text: list[Block] = []
    for block, table in zip(section.blocks, is_table, strict=True):
        if not table:
            text.append(block)
            continue
        lead = [text.pop()] if text and text[-1]["type"] == "paragraph" else []
        if text:
            groups.append(text)
        text = []
        groups.append([*lead, block])
    if text:
        groups.append(text)
    return groups


def _split_long(group: list[Block], sizes: dict[int, int]) -> list[list[Block]]:
    parts: list[list[Block]] = [[]]
    total = 0
    for block in group:
        size = sizes[id(block)]
        if parts[-1] and total + size > MAX_CHILD_CHARS:
            parts.append([])
            total = 0
        parts[-1].append(block)
        total += size
    return parts


def _metadata(source: str, title: str, blocks: list[Block]) -> Metadata:
    pages = sorted({b["pageNumber"] for b in blocks if b.get("pageNumber") is not None})
    return {"source": source, "title": title, "pages": pages}


def chunk(doc: Document) -> list[Parent]:
    blocks: list[Block] = doc["blocks"]
    title_block = next((b for b in blocks if _is_heading(b, 1)), None)
    title = title_block["text"] if title_block else ""
    body = [b for b in blocks if b["type"] != "heading"]

    rendered = render([[b] for b in body])
    sizes = {id(b): len(md) for b, md in zip(body, rendered, strict=True)}
    tables = {
        id(b)
        for b, md in zip(body, rendered, strict=True)
        if b["type"] == "table" and "<table" in md
    }

    plan: list[tuple[_Chapter, list[_Section]]] = []
    for chapter in _chapters(blocks):
        children: list[_Section] = []
        for section in chapter.sections:
            flags = [id(b) in tables for b in section.blocks]
            for group in _child_groups(section, flags):
                if any(id(b) in tables for b in group):
                    children.append(_Section(section.path, group))
                else:
                    children.extend(
                        _Section(section.path, part)
                        for part in _split_long(group, sizes)
                    )
        if children:
            plan.append((chapter, children))

    batches: list[list[Block]] = []
    for chapter, children in plan:
        head = [] if _is_heading(chapter.blocks[0], 1) else chapter.path[:1]
        batches.append([*head, *chapter.blocks])
        batches.extend([*child.path, *child.blocks] for child in children)
    texts = iter(render(batches))

    source = doc["source"]
    return [
        {
            "text": next(texts),
            "metadata": _metadata(source, title, chapter.blocks),
            "children": [
                {"text": next(texts), "metadata": _metadata(source, title, c.blocks)}
                for c in children
            ],
        }
        for chapter, children in plan
    ]
