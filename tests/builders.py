from typing import Any

Block = dict[str, Any]


def p(text: str, page: int = 1, size: float | None = None) -> Block:
    block: Block = {"type": "paragraph", "text": text, "pageNumber": page}
    if size is not None:
        block["style"] = {"fontSize": size}
    return block


def h(text: str, level: int, page: int = 1) -> Block:
    return {"type": "heading", "text": text, "level": level, "pageNumber": page}


def cell(text: str = "", blocks: list[Block] | None = None) -> dict[str, Any]:
    c: dict[str, Any] = {"text": text, "colSpan": 1, "rowSpan": 1}
    if blocks is not None:
        c["blocks"] = blocks
    return c


def table(rows: list[list[dict[str, Any]]], page: int = 1) -> Block:
    return {
        "type": "table",
        "pageNumber": page,
        "table": {"rows": len(rows), "cols": len(rows[0]), "cells": rows},
    }


def grid(rows: int, cols: int, page: int = 1) -> Block:
    return table([[cell(f"r{r}c{c}") for c in range(cols)] for r in range(rows)], page)


def image(page: int = 1) -> Block:
    return {"type": "image", "text": "image_001.png", "pageNumber": page}
