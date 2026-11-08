"""Хранилище метаданных."""
import json
from pathlib import Path


class MetadataStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = {}

    def load(self):
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def set(self, key: str, value):
        self.data[key] = value

    def get(self, key: str, default=None):
        return self.data.get(key, default)
