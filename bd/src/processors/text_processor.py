"""Обработчик текстовых файлов."""
from pathlib import Path

from .base import BaseProcessor


class TextProcessor(BaseProcessor):
    def __init__(self, config: dict):
        self.config = config

    def supported_extensions(self) -> list[str]:
        return ["txt", "md"]

    def extract(self, file_path: Path) -> str:
        for enc in ("utf-8", "cp1251", "latin-1"):
            try:
                return file_path.read_text(encoding=enc)
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
