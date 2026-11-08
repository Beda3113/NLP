"""Семантический поиск."""
from .filters import MetadataFilter


class Searcher:
    def __init__(self, db, encoder):
        self.db = db
        self.encoder = encoder
        self.filter = MetadataFilter()

    def search(self, query: str, top_k: int = 10, filters: dict = None):
        """Поиск по запросу."""
        # Страховка: запрос должен кодироваться query-энкодером
        if not getattr(self.encoder, "for_query", False):
            raise ValueError(
                "Searcher требует GPUEncoder(for_query=True). "
                "Иначе к запросу применится passage-префикс и качество упадёт."
            )

        query_vec = self.encoder.encode([query], show_progress=False)[0]
        scores, indices = self.db.search(query_vec, top_k=top_k * 2)

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            chunk = self.db.get_chunk(int(idx))
            if not chunk:
                continue

            if filters and not self.filter.match(chunk, filters):
                continue

            results.append({
                "score": float(score),
                "index": int(idx),
                **chunk,
            })
            if len(results) >= top_k:
                break

        return results
