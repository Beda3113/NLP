import pytest
from src.vectorization.chunker import TextChunker


def test_chunk_basic():
    cfg = {"processing": {"chunking": {"chunk_size": 100, "overlap": 20, "min_chunk_size": 10}}}
    chunker = TextChunker(cfg)
    text = "Первое предложение. Второе предложение. Третье предложение. Четвертое предложение."
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert all(len(c) >= 10 for c in chunks)


def test_chunk_empty():
    cfg = {"processing": {"chunking": {"chunk_size": 100, "overlap": 20, "min_chunk_size": 10}}}
    chunker = TextChunker(cfg)
    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []
