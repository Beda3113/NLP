"""GPU-энкодер с корректной поддержкой multi-GPU.
Исправления:
- Модель в главном процессе НЕ грузится на GPU, если n_gpu > 1 (избегаем OOM).
- encode_multi_process получает target_devices через pool (исправлена ошибка TypeError).
- Поддержка префиксов e5 (query: / passage:).
- Работает на CPU, если CUDA недоступна.
"""
from __future__ import annotations
import numpy as np
import torch
from .gpu_manager import GPUManager
from .model_loader import ModelLoader


class GPUEncoder:
    """Энкодер с поддержкой multi-GPU через процессный пул."""

    def __init__(self, config: dict, for_query: bool = False):
        self.config = config
        self.for_query = for_query
        vec_cfg = config.get("vectorization", {}) or {}
        self.model_name = vec_cfg.get("model", "intfloat/multilingual-e5-large-instruct")
        self.batch_size = int(vec_cfg.get("batch_size", 32))
        self.normalize = bool(vec_cfg.get("normalize", True))
        self.query_prefix = vec_cfg.get("query_prefix", "")
        self.passage_prefix = vec_cfg.get("passage_prefix", "")

        self.gpu_manager = GPUManager(config)
        self.gpu_manager.setup()
        
        self.n_gpu = torch.cuda.device_count()
        self.device = GPUManager.device()

        # Если multi-GPU — грузим модель только в пуле, в главном процессе — CPU-копия-координатор
        if self.n_gpu > 1:
            self.model = ModelLoader.load(self.model_name, "cpu")
            self._pool = None
        else:
            self.model = ModelLoader.load(self.model_name, self.device)
            self._pool = None

        # Размерность
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()
        
        print(f"[Encoder] Модель: {self.model_name}, dim={self.dimension}, gpus={self.n_gpu}")

    # ------------------------------------------------------------------
    def _get_pool(self):
        if self._pool is None and self.n_gpu > 1:
            print(f"[Encoder] Запуск multi-process pool на {self.n_gpu} GPU...")
            # target_devices указывается здесь, при старте пула
            self._pool = self.model.start_multi_process_pool(
                target_devices=list(range(self.n_gpu))
            )
        return self._pool

    def _apply_prefix(self, texts: list[str]) -> list[str]:
        prefix = self.query_prefix if self.for_query else self.passage_prefix
        if prefix:
            return [prefix + t for t in texts]
        return texts

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        texts = self._apply_prefix(texts)

        if self.n_gpu > 1:
            pool = self._get_pool()
            # В encode_multi_process НЕ передаем target_devices, так как они уже в пуле
            embeddings = self.model.encode_multi_process(
                texts,
                pool,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
            )
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            )

        return embeddings.astype("float32")

    def close(self) -> None:
        if self._pool is not None:
            self.model.stop_multi_process_pool(self._pool)
            self._pool = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
