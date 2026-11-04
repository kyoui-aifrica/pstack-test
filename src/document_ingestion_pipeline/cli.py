import argparse
import sys
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from kordoc import KordocClient
from openai import OpenAI

from . import chunker, embedder, ingester, judge, parser, preprocessor
from .config import ConfigError, ModelConfig, load_model_config

DOCUMENT_SUFFIXES = {".hwp", ".hwpx", ".pdf"}


class NoChunksError(Exception):
    pass


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    arguments = argparse.ArgumentParser(
        prog="ingest", description="samples/의 문서를 파싱해 저장소에 적재합니다"
    )
    arguments.add_argument("--judge", choices=judge.JUDGE_NAMES, default="rule-based")
    arguments.add_argument(
        "--store", choices=tuple(ingester.STORES), default="postgres"
    )
    arguments.add_argument("--samples", type=Path, default=Path("samples"))
    return arguments.parse_args(argv)


def _ingest_document(
    path: Path,
    kordoc: KordocClient,
    heading_judge: judge.Judge,
    config: ModelConfig,
    embedding_client: OpenAI,
    store: ingester.Store,
) -> tuple[int, int]:
    parsed = parser.parse(path, kordoc)
    parents = chunker.chunk(preprocessor.preprocess(parsed, heading_judge))
    if not parents:
        warnings = "; ".join(
            f"{w['code']}: {w['message']}" for w in parsed.get("warnings", [])
        )
        raise NoChunksError(f"청크가 없습니다 (kordoc 경고: {warnings or '없음'})")
    texts = [child["text"] for parent in parents for child in parent["children"]]
    embeddings = embedder.embed_documents(texts, config, embedding_client)
    store.ingest(parsed["source"], parents, embeddings)
    return len(parents), len(texts)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    load_dotenv(override=False)
    try:
        config = load_model_config()
        embedding_client = embedder.embedding_client()
        store = ingester.get_store(args.store, config)
        heading_judge = judge.get_judge(args.judge)
    except ConfigError as error:
        print(f"설정 오류: {error}", file=sys.stderr)
        return 2
    if not args.samples.is_dir():
        print(f"문서 디렉토리가 없습니다: {args.samples}", file=sys.stderr)
        return 2
    paths = sorted(
        p for p in args.samples.iterdir() if p.suffix.lower() in DOCUMENT_SUFFIXES
    )
    failures = 0
    with parser.open_client() as kordoc:
        for path in paths:
            name = unicodedata.normalize("NFC", path.name)
            try:
                counts = _ingest_document(
                    path, kordoc, heading_judge, config, embedding_client, store
                )
            # One bad document must not stop the rest of the batch.
            except Exception as error:
                failures += 1
                print(f"실패 {name}: {type(error).__name__}: {error}")
            else:
                print(f"성공 {name}: 부모 {counts[0]}개, 자식 {counts[1]}개")
    print(f"{len(paths)}개 중 {len(paths) - failures}개 성공, {failures}개 실패")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
