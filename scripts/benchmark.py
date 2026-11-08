"""Бенчмарк производительности."""
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer


def benchmark():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cuda",
    )

    texts = [f"Тестовый документ номер {i} для бенчмарка скорости." for i in range(1000)]

    start = time.time()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    elapsed = time.time() - start

    print(f"\nОбработано: {len(texts)} текстов")
    print(f"Время: {elapsed:.2f} сек")
    print(f"Скорость: {len(texts)/elapsed:.1f} текстов/сек")
    print(f"Размерность: {embeddings.shape}")


if __name__ == "__main__":
    benchmark()
