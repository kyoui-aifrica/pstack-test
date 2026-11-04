"""promptfoo assertion: ordered precision/recall/F1 of predicted vs gold headings.

Also runnable directly for a quick local score without promptfoo:
    uv run python evals/scoring.py --judge rule-based
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from document_ingestion_pipeline.parser import normalize_text

ROOT = Path(__file__).resolve().parent.parent
Heading = tuple[int, str]


def _key(heading: dict[str, Any]) -> Heading:
    return int(heading["level"]), normalize_text(str(heading["text"]))


def _aligned(gold: list[Heading], pred: list[Heading]) -> list[tuple[int, int]]:
    rows, cols = len(gold), len(pred)
    lcs = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            lcs[i][j] = (
                lcs[i + 1][j + 1] + 1
                if gold[i] == pred[j]
                else max(lcs[i + 1][j], lcs[i][j + 1])
            )
    pairs, i, j = [], 0, 0
    while i < rows and j < cols:
        if gold[i] == pred[j]:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif lcs[i + 1][j] >= lcs[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def score(gold: list[dict[str, Any]], pred: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = _aligned([_key(h) for h in gold], [_key(h) for h in pred])
    matched_gold = {i for i, _ in pairs}
    matched_pred = {j for _, j in pairs}
    precision = len(pairs) / len(pred) if pred else 0.0
    recall = len(pairs) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if pairs else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missed": [gold[i] for i in range(len(gold)) if i not in matched_gold],
        "spurious": [pred[j] for j in range(len(pred)) if j not in matched_pred],
    }


def required_f1(path: str) -> float:
    return 0.95 if path.lower().endswith(".pdf") else 1.0


def _label(heading: dict[str, Any]) -> str:
    return f"L{heading['level']} {heading['text']!r}"


def get_assert(output: str, context: dict[str, Any]) -> dict[str, Any]:
    result = score(context["vars"]["headings"], json.loads(output))
    reason = (
        f"P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1']:.3f}"
        f" | missed: {[_label(h) for h in result['missed']]}"
        f" | spurious: {[_label(h) for h in result['spurious']]}"
    )
    threshold = (context.get("config") or {}).get("threshold") or required_f1(
        context["vars"]["path"]
    )
    return {
        "pass": result["f1"] >= threshold,
        "score": result["f1"],
        "reason": reason,
        "namedScores": {k: result[k] for k in ("precision", "recall", "f1")},
    }


def main() -> int:
    from heading_provider import predict_headings

    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default="rule-based")
    parser.add_argument("--cases", default=str(ROOT / "evals/tests/test_cases.yaml"))
    args = parser.parse_args()
    cases = yaml.safe_load(Path(args.cases).read_text(encoding="utf-8"))
    for case in cases:
        pred = predict_headings(ROOT / case["vars"]["path"], args.judge)
        result = score(case["vars"]["headings"], pred)
        print(
            f"F1={result['f1']:.3f} P={result['precision']:.3f} "
            f"R={result['recall']:.3f}  {case['description']}"
        )
        for h in result["missed"]:
            print(f"    missed   {_label(h)}")
        for h in result["spurious"]:
            print(f"    spurious {_label(h)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    sys.exit(main())
