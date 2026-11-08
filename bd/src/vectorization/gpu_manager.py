"""Управление GPU. Не падает на CPU-машинах (allow_cpu)."""
from __future__ import annotations

import os

import torch


class GPUManager:
    """Менеджер GPU-ресурсов."""

    def __init__(self, config: dict):
        self.config = config
        self.gpu_cfg = config.get("gpu", {}) or {}
        self.allow_cpu = bool(self.gpu_cfg.get("allow_cpu", True))

    def setup(self) -> None:
        if not torch.cuda.is_available():
            if self.allow_cpu:
                print("[GPU] CUDA недоступна — работаем на CPU")
                return
            raise RuntimeError(
                "CUDA недоступна. Проверьте драйверы NVIDIA или включите gpu.allow_cpu: true"
            )

        # CUDA_VISIBLE_DEVICES уже выставлен в _bootstrap до import torch.
        # Здесь только фиксируем устройство по умолчанию (для одно-GPU режима).
        devices = self.gpu_cfg.get("devices") or [0]
        if len(devices) == 1 and devices[0] < torch.cuda.device_count():
            torch.cuda.set_device(devices[0])

        torch.cuda.empty_cache()
        n_gpu = torch.cuda.device_count()
        print(f"[GPU] Видимых GPU: {n_gpu}")
        for i in range(n_gpu):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  [{i}] {name} — {mem:.1f} GB")

    def n_gpu(self) -> int:
        return torch.cuda.device_count()

    @staticmethod
    def device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
