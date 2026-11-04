import contextlib
import unicodedata

import pytest

from document_ingestion_pipeline import (
    chunker,
    cli,
    embedder,
    ingester,
    parser,
    preprocessor,
)

SCAN_WARNING = {"code": "NEEDS_OCR", "message": "텍스트가 없는 스캔 PDF", "page": 1}


class RecordingStore:
    def __init__(self) -> None:
        self.ingested: dict[str, tuple[list, list]] = {}

    def ingest(self, source, parents, embeddings):
        self.ingested[source] = (parents, embeddings)

    def count(self, source):
        parents, embeddings = self.ingested[source]
        return len(parents), len(embeddings)


def fake_parse(path, client):
    source = unicodedata.normalize("NFC", path.name)
    if source == "스캔.pdf":
        return {"source": source, "blocks": [], "warnings": [SCAN_WARNING]}
    return {"source": source, "blocks": ["본문"]}


def fake_chunk(doc):
    metadata = {"source": doc["source"], "title": "공고", "pages": [1]}
    return [
        {
            "text": f"# 공고\n{block}",
            "metadata": metadata,
            "children": [
                {"text": f"# 공고\n{block} 1", "metadata": metadata},
                {"text": f"# 공고\n{block} 2", "metadata": metadata},
            ],
        }
        for block in doc["blocks"]
    ]


@pytest.fixture
def workspace(tmp_path, monkeypatch, fake_embedding_api):
    (tmp_path / "config.yaml").write_text(
        "models:\n  m:\n    index_name: m\n    dimension: 4\n", encoding="utf-8"
    )
    samples = tmp_path / "samples"
    samples.mkdir()
    for name in ("공고.hwp", "스캔.pdf", "안내서.HWPX", "메모.txt"):
        (samples / unicodedata.normalize("NFD", name)).write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EMBEDDING_MODEL", "m")

    store = RecordingStore()
    parsed: list[str] = []

    def recording_parse(path, client):
        parsed.append(unicodedata.normalize("NFC", path.name))
        return fake_parse(path, client)

    monkeypatch.setattr(parser, "open_client", contextlib.nullcontext)
    monkeypatch.setattr(parser, "parse", recording_parse)
    monkeypatch.setattr(preprocessor, "preprocess", lambda doc, judge: doc)
    monkeypatch.setattr(chunker, "chunk", fake_chunk)
    monkeypatch.setattr(embedder, "embedding_client", fake_embedding_api.client)
    monkeypatch.setattr(ingester, "get_store", lambda name, config: store)
    return {"store": store, "parsed": parsed, "embedding": fake_embedding_api}


def test_document_without_chunks_fails_with_kordoc_warning_while_others_succeed(
    workspace, capsys
):
    exit_code = cli.main([])

    out = capsys.readouterr().out
    assert exit_code != 0
    failure = next(line for line in out.splitlines() if "스캔.pdf" in line)
    assert "실패" in failure and "NEEDS_OCR" in failure
    assert set(workspace["store"].ingested) == {"공고.hwp", "안내서.HWPX"}
    assert workspace["store"].count("공고.hwp") == (1, 2)


def test_only_hwp_hwpx_and_pdf_files_are_processed(workspace):
    cli.main([])

    assert sorted(workspace["parsed"]) == ["공고.hwp", "스캔.pdf", "안내서.HWPX"]


def test_each_document_embeds_its_children_in_one_request(workspace):
    cli.main([])

    assert [len(r["input"]) for r in workspace["embedding"].requests] == [2, 2]


def test_all_documents_succeeding_exits_zero(workspace, capsys):
    (workspace_samples := cli.Path("samples")).joinpath(
        unicodedata.normalize("NFD", "스캔.pdf")
    ).unlink()

    assert cli.main(["--samples", str(workspace_samples)]) == 0
    assert "실패 " not in capsys.readouterr().out


def test_failure_in_one_document_does_not_stop_the_next(workspace, monkeypatch, capsys):
    def flaky_parse(path, client):
        if unicodedata.normalize("NFC", path.name) == "공고.hwp":
            raise parser.ParseError("손상된 파일")
        return fake_parse(path, client)

    monkeypatch.setattr(parser, "parse", flaky_parse)

    assert cli.main([]) != 0
    assert "손상된 파일" in capsys.readouterr().out
    assert "안내서.HWPX" in workspace["store"].ingested


def test_unknown_model_stops_before_any_document_is_parsed(
    workspace, monkeypatch, capsys
):
    monkeypatch.setenv("EMBEDDING_MODEL", "no/such-model")

    assert cli.main([]) == 2
    assert "no/such-model" in capsys.readouterr().err
    assert workspace["parsed"] == []


def test_unknown_judge_is_rejected_by_the_argument_parser(workspace):
    with pytest.raises(SystemExit):
        cli.main(["--judge", "magic"])
