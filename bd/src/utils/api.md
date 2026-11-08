# API

## CLI
- `docvec index --input DIR --output DIR` — индексация
- `docvec search --db DIR --query TEXT` — поиск
- `docvec info --db DIR` — информация о базе

## Python
```python
from src.vectorization.encoder import GPUEncoder
from src.storage.vector_db import VectorDB
from src.search.searcher import Searcher

encoder = GPUEncoder(config)
db = VectorDB(path, config)
db.load()
searcher = Searcher(db, encoder)
results = searcher.search("запрос", top_k=10)
