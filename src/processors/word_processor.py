"""Обработчик Word с сохранением заголовков."""
from pathlib import Path

from .base import BaseProcessor


class WordProcessor(BaseProcessor):
    def __init__(self, config: dict):
        self.config = config

    def supported_extensions(self) -> list[str]:
        return ["docx"]

    def extract(self, file_path: Path) -> str:
        if file_path.name.startswith("~$"):
            return ""
        try:
            from docx import Document
            doc = Document(file_path)
        except Exception:
            return ""

        parts: list[str] = []
        for para in doc.paragraphs:
            if not para.text.strip():
                continue
            style = (para.style.name or "").lower()
            if style.startswith("heading"):
                parts.append(f"\n## {para.text.strip()}")
            else:
                parts.append(para.text)

        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        return "\n".join(parts)
