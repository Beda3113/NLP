"""OCR обработчик изображений."""
from pathlib import Path
import pytesseract
from PIL import Image
from .base import BaseProcessor


class ImageProcessor(BaseProcessor):
    def __init__(self, config: dict):
        self.config = config
        ocr_cfg = config.get("processing", {}).get("ocr", {})
        self.lang = ocr_cfg.get("languages", "rus+eng")
        self.psm = ocr_cfg.get("psm", 6)

    def supported_extensions(self) -> list:
        return ["jpg", "jpeg", "png", "tiff", "bmp"]

    def extract(self, file_path: Path) -> str:
        """
        Извлекает текст из изображения с помощью Tesseract OCR.
        Возвращает пустую строку в случае ошибки.
        """
        try:
            # Открываем изображение
            img = Image.open(file_path)
            
            # Конвертируем в RGB, если необходимо (например, для RGBA png)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Настраиваем параметры Tesseract
            # PSM 6 — Assume a single uniform block of text.
            custom_config = f"--oem 3 --psm {self.psm}"
            
            # Выполняем распознавание
            text = pytesseract.image_to_string(
                img, 
                lang=self.lang, 
                config=custom_config
            )
            
            return text.strip()
        except Exception as e:
            # Логируем ошибку в реальном проекте, здесь просто возвращаем пустоту
            # print(f"OCR Error for {file_path}: {e}")
            return ""
