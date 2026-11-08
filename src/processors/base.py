"""Базовый класс процессора."""
from abc import ABC, abstractmethod
from pathlib import Path


class BaseProcessor(ABC):
    """Абстрактный процессор. extract() возвращает ТОЛЬКО текст (str).

    Метаданные процессор может сохранять в self.meta, если нужны
    вызывающему (в текущей версии registry их не прокидывает).
    """

    @abstractmethod
    def supported_extensions(self) -> list[str]: ...

    @abstractmethod
    def extract(self, file_path: Path) -> str: ...

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower().lstrip(".") in self.supported_extensions()
