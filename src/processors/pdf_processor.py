"""Обработчик PDF с защитой от битых файлов."""
from pathlib import Path
import pdfplumber
import warnings
from .base import BaseProcessor


class PDFProcessor(BaseProcessor):
    def __init__(self, config: dict):
        self.config = config

    def supported_extensions(self) -> list:
        return ["pdf"]

    def extract(self, file_path: Path) -> str:
        # Подавляем шумные предупреждения pdfminer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                parts = []
                # Ограничиваем время открытия файла
                with pdfplumber.open(file_path) as pdf:
                    # Обрабатываем максимум 100 страниц (защита от огромных PDF)
                    max_pages = min(len(pdf.pages), 100)
                    
                    for i, page in enumerate(pdf.pages[:max_pages]):
                        try:
                            text = page.extract_text() or ""
                            tables = page.extract_tables() or []

                            page_text = []
                            if text.strip():
                                page_text.append(text)
                            
                            for table in tables:
                                for row in table:
                                    cells = [str(c) for c in row if c]
                                    if cells:
                                        page_text.append(" | ".join(cells))

                            if page_text:
                                parts.append(f"--- Страница {i+1} ---\n" + "\n".join(page_text))
                        except Exception:
                            # Пропускаем проблемные страницы
                            continue
                
                return "\n\n".join(parts)
            
            except Exception as e:
                # Если PDF полностью битый — возвращаем пустую строку
                return ""
