"""Кэш обработки (для инкрементальных обновлений)."""
import hashlib
import json
from pathlib import Path


class ProcessingCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.cache = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    @staticmethod
    def file_hash(file_path: Path) -> str:
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def is_changed(self, file_path: Path) -> bool:
        key = str(file_path)
        current_hash = self.file_hash(file_path)
        return self.cache.get(key) != current_hash

    def mark(self, file_path: Path):
        self.cache[str(file_path)] = self.file_hash(file_path)
