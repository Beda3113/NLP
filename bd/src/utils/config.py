"""Загрузка конфигурации с deep-merge дефолта и пользовательского файла."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

_DEFAULTS: dict[str, Any] = {
    "paths": {"vector_db": "./vector_db", "config": "config/default.yaml"},
    "processing": {
        "supported_formats": "xlsx,xls,csv,pdf,jpg,jpeg,png,tiff,bmp,docx,txt,md",
        "ignored_formats": "mp4,mov,mpg,mpeg,avi,mkv,wmv,flv,webm,mp3,wav,flac,aac,ogg,wma,m4a,exe,msi,zip,rar,7z,tar,gz",
        "timeout_per_file": 30,
        "workers": None,
        "ocr": {"languages": "rus+eng", "psm": 6, "preprocess": True, "low_text_threshold": 50},
        "chunking": {
            "chunk_size": 512, "overlap_sentences": 2, "min_chunk_size": 20,
            "protect_abbreviations": True, "table_atomic": True,
        },
    },
    "vectorization": {
        "model": "intfloat/multilingual-e5-large-instruct",
        "batch_size": 32, "normalize": True,
        "use_reranker": True, "reranker_model": "BAAI/bge-reranker-v2-m3",
        "query_prefix": "query: ", "passage_prefix": "passage: ",
    },
    "gpu": {"use_multi_gpu": True, "devices": [0], "allow_cpu": True, "cache_visible_devices": True},
    "storage": {"backend": "faiss", "index_type": "IndexFlatIP"},
    "search": {
        "hybrid": True, "dense_weight": 0.7, "bm25_weight": 0.3,
        "use_mmr": True, "mmr_lambda": 0.5, "max_chunks_per_source": 3,
        "candidates_multiplier": 5,
    },
    "domain": {"abbreviations": {}, "synonyms": {}, "units": []},
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str | None = None) -> dict:
    """Загружает дефолт, мержит с пользовательским файлом (если есть)."""
    cfg = copy.deepcopy(_DEFAULTS)

    # Сначала дефолтный yaml из пакета (если есть)
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    # Затем пользовательский
    if path:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg
