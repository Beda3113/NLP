"""Чанкер для технических текстов."""
from __future__ import annotations

import re

from ..domain.normalizer import split_sentences

_TABLE_LINE = re.compile(r"^\s*\S.*\|\s*\S")


class TextChunker:
    def __init__(self, config: dict):
        cfg = (config.get("processing", {}) or {}).get("chunking", {}) or {}
        self.chunk_size = int(cfg.get("chunk_size", 512))
        self.overlap_sentences = int(cfg.get("overlap_sentences", 2))
        self.min_size = int(cfg.get("min_chunk_size", 20))
        self.protect_abbrev = bool(cfg.get("protect_abbreviations", True))
        self.table_atomic = bool(cfg.get("table_atomic", True))

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # 1. Разделяем на блоки: таблицы / не-таблицы
        blocks = self._split_blocks(text)

        chunks: list[str] = []
        for block_type, block_text in blocks:
            if block_type == "table" and self.table_atomic:
                # Таблица — атомарна: либо целиком, либо по строкам с заголовком
                chunks.extend(self._chunk_table(block_text))
            else:
                chunks.extend(self._chunk_text(block_text))

        return [c for c in chunks if len(c) >= self.min_size]

    # ------------------------------------------------------------------
    def _split_blocks(self, text: str) -> list[tuple[str, str]]:
        """Делит текст на блоки: ('table', ...) и ('text', ...)."""
        lines = text.split("\n")
        blocks: list[tuple[str, list[str]]] = []
        current_type, current_lines = "text", []

        for line in lines:
            is_table = bool(_TABLE_LINE.match(line))
            line_type = "table" if is_table else "text"
            if line_type != current_type and current_lines:
                blocks.append((current_type, current_lines))
                current_lines = []
            current_type = line_type
            current_lines.append(line)
        if current_lines:
            blocks.append((current_type, current_lines))

        return [(t, "\n".join(ls)) for t, ls in blocks if any(l.strip() for l in ls)]

    def _chunk_table(self, table_text: str) -> list[str]:
        """Таблица сохраняется целиком; если очень большая — с заголовком в каждом чанке."""
        lines = [l for l in table_text.split("\n") if l.strip()]
        if not lines:
            return []
        header = lines[0]
        if len(table_text) <= self.chunk_size * 2:
            return [table_text]

        chunks, current = [], [header]
        current_len = len(header)
        for line in lines[1:]:
            if current_len + len(line) > self.chunk_size and len(current) > 1:
                chunks.append("\n".join(current))
                current = [header, line]  # заголовок повторяем
                current_len = len(header) + len(line)
            else:
                current.append(line)
                current_len += len(line)
        if len(current) > 1:
            chunks.append("\n".join(current))
        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        text = re.sub(r"[ \t]+", " ", text).strip()
        if not text:
            return []
        sentences = split_sentences(text, protect_abbrev=self.protect_abbrev)

        chunks, current = [], []
        current_len = 0

        for sent in sentences:
            slen = len(sent)
            if current_len + slen > self.chunk_size and current:
                chunks.append(" ".join(current))
                # overlap по предложениям, а не по символам
                overlap = current[-self.overlap_sentences:] if self.overlap_sentences > 0 else []
                current = list(overlap)
                current_len = sum(len(s) for s in current)
            current.append(sent)
            current_len += slen

        if current:
            chunks.append(" ".join(current))
        return chunks
