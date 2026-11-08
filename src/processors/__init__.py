"""Инициализация модуля процессоров."""
from .registry import ProcessorRegistry
from .base import BaseProcessor

__all__ = ["ProcessorRegistry", "BaseProcessor"]
