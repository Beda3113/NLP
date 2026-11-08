"""Импортируется ПЕРВЫМ в main.py.

Выставляет CUDA_VISIBLE_DEVICES до импорта torch, иначе
torch.cuda.device_count() закэширует видимость при первой инициализации.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_config_path() -> Path:
    cfg_path = Path("config/default.yaml")
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            cfg_path = Path(args[i + 1])
        elif arg.startswith("--config="):
            cfg_path = Path(arg.split("=", 1)[1])
    return cfg_path


def _load_gpu_devices_early() -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return  # пользователь уже задал явно

    cfg_path = _resolve_config_path()
    if not cfg_path.exists():
        return

    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return

    gpu = (cfg.get("gpu") or {})
    if not gpu.get("cache_visible_devices", True):
        return

    devices = gpu.get("devices") or []
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)


_load_gpu_devices_early()
