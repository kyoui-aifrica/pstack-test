import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

CONFIG_PATH = Path("config.yaml")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    index_name: str
    dimension: int
    document_prefix: str = ""


_KEYS = {f.name for f in fields(ModelConfig)} - {"name"}
_REQUIRED = {"index_name", "dimension"}


def _model_config(name: str, entry: object) -> ModelConfig:
    if not isinstance(entry, dict):
        raise ConfigError(f"모델 '{name}' 설정이 매핑이 아닙니다")
    if unknown := entry.keys() - _KEYS:
        raise ConfigError(
            f"모델 '{name}'에 알 수 없는 키가 있습니다: {sorted(unknown)}"
        )
    if missing := _REQUIRED - entry.keys():
        raise ConfigError(f"모델 '{name}'에 필수 키가 없습니다: {sorted(missing)}")
    dimension = entry["dimension"]
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ConfigError(
            f"모델 '{name}'의 dimension은 양의 정수여야 합니다: {dimension!r}"
        )
    if not isinstance(entry["index_name"], str) or not entry["index_name"]:
        raise ConfigError(f"모델 '{name}'의 index_name은 빈 문자열일 수 없습니다")
    if not isinstance(entry.get("document_prefix", ""), str):
        raise ConfigError(f"모델 '{name}'의 document_prefix는 문자열이어야 합니다")
    return ModelConfig(name=name, **entry)


def load_model_config(
    name: str | None = None, path: str | os.PathLike[str] | None = None
) -> ModelConfig:
    name = name or os.environ.get("EMBEDDING_MODEL")
    if not name:
        raise ConfigError("EMBEDDING_MODEL이 설정되지 않았습니다")
    config_path = Path(path) if path is not None else CONFIG_PATH
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"설정 파일을 읽을 수 없습니다: {config_path}") from error
    if not isinstance(raw, dict) or set(raw) != {"models"}:
        raise ConfigError(f"{config_path}의 최상위 키는 'models' 하나여야 합니다")
    if not isinstance(raw["models"], dict):
        raise ConfigError(f"{config_path}의 'models'가 매핑이 아닙니다")
    models = {key: _model_config(key, entry) for key, entry in raw["models"].items()}
    if name not in models:
        raise ConfigError(
            f"설정에 없는 임베딩 모델입니다: {name} (가능: {sorted(models)})"
        )
    return models[name]
