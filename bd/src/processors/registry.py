"""Реестр процессоров."""
from pathlib import Path
from .excel_processor import ExcelProcessor
from .pdf_processor import PDFProcessor
from .image_processor import ImageProcessor
from .word_processor import WordProcessor
from .text_processor import TextProcessor


class ProcessorRegistry:
    """Реестр всех процессоров."""

    def __init__(self, config: dict):
        self.config = config
        self.processors = [
            ExcelProcessor(config),
            PDFProcessor(config),
            ImageProcessor(config),
            WordProcessor(config),
            TextProcessor(config),
        ]
        self._ext_map = {}
        for p in self.processors:
            for ext in p.supported_extensions():
                self._ext_map[ext] = p

    def process(self, file_path: Path) -> str:
        """Обработка файла соответствующим процессором."""
        ext = file_path.suffix.lower().lstrip(".")
        processor = self._ext_map.get(ext)
        if not processor:
            raise ValueError(f"Нет процессора для формата: {ext}")
        return processor.extract(file_path)
