"""Фильтры по метаданным."""
class MetadataFilter:
    def match(self, chunk: dict, filters: dict) -> bool:
        for key, value in filters.items():
            if key == "source_contains":
                if value not in chunk.get("source", ""):
                    return False
            elif key == "extension":
                source = chunk.get("source", "")
                if not source.endswith(f".{value}"):
                    return False
            else:
                if chunk.get(key) != value:
                    return False
        return True
