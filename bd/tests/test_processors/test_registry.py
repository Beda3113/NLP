from pathlib import Path
from src.processors.registry import ProcessorRegistry


def test_registry_handles_known_extensions():
    cfg = {"processing": {"ocr": {"languages": "rus+eng", "psm": 6}}}
    reg = ProcessorRegistry(cfg)
    assert reg.process(Path("test.txt")) == ""
