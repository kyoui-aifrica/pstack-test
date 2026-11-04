import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

from kordoc import KordocClient

Block = dict[str, Any]
Document = dict[str, Any]

KORDOC_VERSION = "4.24.11"
RENDER_SCRIPT = Path(__file__).with_name("render.mjs")
TABLE_TAG_PATTERN = r"</?(?:table|thead|tbody|tr|th|td)\b[^>]*>|<br\s*/?>"

_RENDERER_TAG = re.compile(TABLE_TAG_PATTERN + r"|</?(?:u|sup|sub)>", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


class ParseError(Exception):
    pass


def _engine_entry() -> str:
    if cli := os.environ.get("KORDOC_CLI"):
        return cli
    for root in (Path(__file__).resolve(), Path.cwd() / "_"):
        for parent in root.parents:
            entry = parent / "node_modules" / "kordoc" / "dist" / "cli.js"
            if entry.is_file():
                return str(entry)
    raise ParseError(
        "kordoc 엔진이 없습니다. 프로젝트 루트에서 `npm install`을 실행하세요"
    )


def open_client() -> KordocClient:
    return KordocClient(cli=_engine_entry())


def parse(path: str | os.PathLike[str], client: KordocClient) -> Document:
    result = client.parse(path, html_tables=True, images=False)
    if not result.success:
        raise ParseError(str(result.raw.get("error") or "kordoc 파싱 실패"))
    source = unicodedata.normalize("NFC", Path(path).name)
    return {**result.raw, "source": source}


def render(batches: list[list[Block]]) -> list[str]:
    if not batches:
        return []
    proc = subprocess.run(
        ["node", str(RENDER_SCRIPT)],
        input=json.dumps(batches, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ParseError(f"kordoc 렌더링 실패: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def block_text(block: Block) -> str:
    if block["type"] != "table":
        return block.get("text", "")
    cells = (cell for row in block["table"]["cells"] for cell in row if cell)
    return " ".join(t for cell in cells if (t := cell.get("text", "").strip()))


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub("", _RENDERER_TAG.sub("", text))
