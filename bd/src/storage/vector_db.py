"""FAISS векторная база."""
import json
import numpy as np
import faiss
from pathlib import Path


class VectorDB:
    def __init__(self, path: Path, config: dict, dimension: int = None):
        self.path = Path(path)
        self.config = config
        # Если размерность не передана явно, пытаемся взять из конфига модели
        if dimension is None:
            vec_cfg = config.get("vectorization", {})
            model_name = vec_cfg.get("model", "")
            # Эвристика для популярных моделей, если размерность неизвестна
            if "large" in model_name or "e5-large" in model_name:
                self.dimension = 1024
            elif "base" in model_name:
                self.dimension = 768
            else:
                self.dimension = 384  # Default for MiniLM
        else:
            self.dimension = dimension
            
        self.index = None
        self.metadata = []
        self.processed_files = set()
        
        storage_cfg = config.get("storage", {})
        self.index_type = storage_cfg.get("index_type", "IndexFlatIP")

    def load_or_create(self):
        """Загрузка существующей или создание новой базы."""
        if (self.path / "index.faiss").exists():
            self.load()
            # Проверка совместимости размерности при загрузке
            if self.index and self.index.d != self.dimension:
                print(f"[WARNING] Размерность индекса ({self.index.d}) не совпадает с текущей ({self.dimension}). Пересоздаем базу.")
                self._create_new()
        else:
            self._create_new()

    def _create_new(self):
        """Создание нового индекса."""
        self.path.mkdir(parents=True, exist_ok=True)
        print(f"[DB] Создание нового индекса: type={self.index_type}, dim={self.dimension}")
        
        if self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            # По умолчанию используем Inner Product (косинусное сходство при нормализации)
            self.index = faiss.IndexFlatIP(self.dimension)
            
        self.metadata = []
        self.processed_files = set()

    def load(self):
        """Загрузка с диска."""
        self.index = faiss.read_index(str(self.path / "index.faiss"))
        self.dimension = self.index.d  # Обновляем размерность из файла
        
        meta_path = self.path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.metadata = data.get("chunks", [])
            self.processed_files = set(data.get("processed_files", []))

    def save(self):
        """Сохранение на диск."""
        self.path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.path / "index.faiss"))
        with open(self.path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "dimension": self.dimension,
                "chunks": self.metadata,
                "processed_files": list(self.processed_files),
            }, f, ensure_ascii=False, indent=2)

    def add(self, embeddings: np.ndarray, chunks: list):
        """Добавление векторов и метаданных."""
        if self.index is None:
            self._create_new()
            
        # Приводим к float32, так как FAISS требует именно этот тип
        embeddings = embeddings.astype("float32")
        
        # Проверка размерности перед добавлением
        if embeddings.shape[1] != self.index.d:
            raise ValueError(f"Размерность векторов ({embeddings.shape[1]}) не совпадает с индексом ({self.index.d})")
            
        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 10):
        """Поиск по индексу."""
        if self.index is None or self.index.ntotal == 0:
            return [], []
        
        query_vec = query_vec.reshape(1, -1).astype("float32")
        # Проверка размерности запроса
        if query_vec.shape[1] != self.index.d:
             raise ValueError(f"Размерность запроса ({query_vec.shape[1]}) не совпадает с индексом ({self.index.d})")
             
        scores, indices = self.index.search(query_vec, top_k)
        return scores[0], indices[0]

    def count(self) -> int:
        return self.index.ntotal if self.index else 0

    def is_processed(self, file_path: Path) -> bool:
        return str(file_path) in self.processed_files

    def mark_processed(self, file_path: Path):
        self.processed_files.add(str(file_path))

    def processed_count(self) -> int:
        return len(self.processed_files)

    def get_chunk(self, idx: int) -> dict:
        if 0 <= idx < len(self.metadata):
            return self.metadata[idx]
        return {}
