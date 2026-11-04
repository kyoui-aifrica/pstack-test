"""promptfoo provider: parse and preprocess a document, return its headings as JSON."""

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from document_ingestion_pipeline.judge import get_judge
from document_ingestion_pipeline.parser import open_client, parse
from document_ingestion_pipeline.preprocessor import preprocess

ROOT = Path(__file__).resolve().parent.parent


def predict_headings(path: Path, judge_name: str) -> list[dict[str, Any]]:
    load_dotenv(ROOT / ".env")
    with open_client() as client:
        parsed = parse(path, client)
    doc = preprocess(parsed, get_judge(judge_name))
    return [{"level": h["level"], "text": h["text"]} for h in doc["outline"]]


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]):
    judge_name = options.get("config", {}).get("judge", "rule-based")
    path = ROOT / context["vars"]["path"]
    try:
        headings = predict_headings(path, judge_name)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"output": json.dumps(headings, ensure_ascii=False)}
