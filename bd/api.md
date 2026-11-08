# API

## CLI
- `docvec index --input DIR --output DIR` — индексация
- `docvec search --db DIR --query TEXT` — поиск
- `docvec info --db DIR` — информация о базе
- `docvec export --db DIR --query TEXT --output result.json` — экспорт в JSON

## Python

```python
from src.utils.config import load_config
from src.vectorization.encoder import GPUEncoder
from src.storage.vector_db import VectorDB
from src.search.searcher import Searcher

cfg = load_config("config/default.yaml")
cfg.setdefault("gpu", {})["use_multi_gpu"] = False
cfg["gpu"]["allow_cpu"] = True

encoder = GPUEncoder(cfg, for_query=True)
db = VectorDB(path="vector_db", config=cfg)
db.load()
searcher = Searcher(db, encoder)
results = searcher.search("запрос", top_k=10)
encoder.close()
```
