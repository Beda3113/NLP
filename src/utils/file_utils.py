"""Утилиты для работы с файлами."""
from pathlib import Path
from typing import List

HARD_IGNORED = {
    "mp4", "mov", "mpg", "mpeg", "avi", "mkv", "wmv", "flv", "webm", "m4v", "3gp", "ogv",
    "mp3", "wav", "flac", "aac", "ogg", "wma", "m4a",
    "exe", "msi", "deb", "rpm", "appimage", "dmg",
    "zip", "rar", "7z", "tar", "gz", "bz2", "xz",
    "tmp", "temp", "bak", "old", "lnk",
}


def _parse_exts(s: str | None) -> set[str]:
    if not s:
        return set()
    # ← ИСПРАВЛЕНО: split по запятой, а не итерация по символам
    return {e.strip().lower().lstrip(".") for e in s.split(",") if e.strip()}


def scan_files(directory: Path, formats_str: str, ignored_formats: str | None = None) -> List[Path]:
    """Рекурсивный поиск файлов поддерживаемых форматов."""
    extensions = _parse_exts(formats_str)
    ignored = _parse_exts(ignored_formats) | HARD_IGNORED
    extensions -= ignored

    files: list[Path] = []
    skipped = 0
    for path in Path(directory).rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("~$"):
            skipped += 1
            continue
        ext = path.suffix.lower().lstrip(".")
        if ext in ignored:
            skipped += 1
            continue
        if ext in extensions:
            files.append(path)
    if skipped:
        print(f"[Scan] Пропущено файлов (видео/аудио/архивы/временные): {skipped}")
    return sorted(files)
