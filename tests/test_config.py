from pathlib import Path

import pytest

from document_ingestion_pipeline.config import (
    ConfigError,
    ModelConfig,
    load_model_config,
)

REPO_CONFIG = Path(__file__).parents[1] / "config.yaml"


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_named_model_loads_index_dimension_and_prefix():
    config = load_model_config("google/embeddinggemma-2", REPO_CONFIG)

    assert config == ModelConfig(
        name="google/embeddinggemma-2",
        index_name="embeddinggemma-2",
        dimension=768,
        document_prefix="title: {title} | text: ",
    )


def test_model_without_prefix_gets_empty_prefix():
    config = load_model_config("nlpai-lab/KURE-v1", REPO_CONFIG)

    assert config == ModelConfig("nlpai-lab/KURE-v1", "kure-v1", 1024, "")


def test_model_is_chosen_by_embedding_model_variable(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    assert load_model_config(path=REPO_CONFIG).index_name == "bge-m3"


def test_config_is_read_from_working_directory_by_default(monkeypatch, tmp_path):
    write_config(tmp_path, "models:\n  m:\n    index_name: m-index\n    dimension: 3\n")
    monkeypatch.chdir(tmp_path)

    assert load_model_config("m") == ModelConfig("m", "m-index", 3)


def test_unknown_model_raises_immediately():
    with pytest.raises(ConfigError, match="no/such-model"):
        load_model_config("no/such-model", REPO_CONFIG)


def test_missing_embedding_model_variable_raises(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    with pytest.raises(ConfigError, match="EMBEDDING_MODEL"):
        load_model_config(path=REPO_CONFIG)


def test_unknown_key_raises(tmp_path):
    path = write_config(
        tmp_path, "models:\n  m:\n    index_name: m\n    dimension: 3\n    dim: 3\n"
    )

    with pytest.raises(ConfigError, match="dim"):
        load_model_config("m", path)


def test_unknown_key_in_another_model_also_raises(tmp_path):
    path = write_config(
        tmp_path,
        "models:\n"
        "  good:\n    index_name: good\n    dimension: 3\n"
        "  bad:\n    index_name: bad\n    dimension: 3\n    prefix: x\n",
    )

    with pytest.raises(ConfigError, match="prefix"):
        load_model_config("good", path)


def test_missing_required_key_raises(tmp_path):
    path = write_config(tmp_path, "models:\n  m:\n    index_name: m\n")

    with pytest.raises(ConfigError, match="dimension"):
        load_model_config("m", path)


@pytest.mark.parametrize("value", ["'1024'", "0", "-1", "true"])
def test_dimension_that_is_not_a_positive_integer_raises(tmp_path, value):
    path = write_config(
        tmp_path, f"models:\n  m:\n    index_name: m\n    dimension: {value}\n"
    )

    with pytest.raises(ConfigError, match="dimension"):
        load_model_config("m", path)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="config.yaml"):
        load_model_config("m", tmp_path / "config.yaml")
