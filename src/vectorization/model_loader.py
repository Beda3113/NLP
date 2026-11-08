"""Загрузка моделей."""
from sentence_transformers import SentenceTransformer


class ModelLoader:
    @staticmethod
    def load(model_name: str, device: str) -> SentenceTransformer:
        return SentenceTransformer(model_name, device=device)
