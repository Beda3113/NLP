# src/domain/normalizer.py
"""Нормализация технических текстов: аббревиатуры, синонимы, query expansion."""
from __future__ import annotations

import re
from typing import Iterable

# Защита от ложных разрезов предложений
_ABBREV_DOT = re.compile(
    r"\b(?:т\.\s*[еик]|см|рис|табл|стр|г|гг|в|вв|мм|см|м|км|кг|"
    r"мА|кВ|кВт|Гц|ГОСТ|ТУ|ТЗ|РЭ|ЭД|АСУ|ТП|КИП|ЧПУ|ПЛК|др|пр)\.",
    re.IGNORECASE,
)
_NUM_DOT = re.compile(r"(\d)\.(\d)")
_CODE_DASH = re.compile(r"\b([A-ZА-Я]{1,8})-(\d+)\b")
_PROTECT_DOT = "\u2024"
_PROTECT_DASH = "\u2011"


def protect(text: str) -> str:
    """Заменяет «опасные» точки/дефисы на плейсхолдеры, чтобы не резать предложения."""
    text = _ABBREV_DOT.sub(lambda m: m.group(0).replace(".", _PROTECT_DOT), text)
    text = _NUM_DOT.sub(rf"\1{_PROTECT_DOT}\2", text)
    text = _CODE_DASH.sub(rf"\1{_PROTECT_DASH}\2", text)
    return text


def restore(text: str) -> str:
    return text.replace(_PROTECT_DOT, ".").replace(_PROTECT_DASH, "-")


def split_sentences(text: str, protect_abbrev: bool = True) -> list[str]:
    """Разбивает текст на предложения, не ломая аббревиатуры/числа/коды."""
    if protect_abbrev:
        text = protect(text)
    # Режем только по .!? + пробел + заглавная (рус/лат)
    parts = re.split(r"(?<=[.!?])\s+(?=[А-ЯA-ZЁ])", text)
    return [restore(p).strip() for p in parts if p.strip()]


def expand_query(query: str, cfg: dict) -> list[str]:
    """Возвращает варианты запроса с раскрытыми аббревиатурами и синонимами."""
    variants: list[str] = [query]
    domain = cfg.get("domain", {})
    for abbr, expansions in (domain.get("abbreviations") or {}).items():
        if re.search(rf"\b{re.escape(abbr)}\b", query):
            for exp in expansions:
                variants.append(re.sub(rf"\b{re.escape(abbr)}\b", exp, query))
    for term, synonyms in (domain.get("synonyms") or {}).items():
        if re.search(rf"\b{re.escape(term)}\b", query, re.IGNORECASE):
            for syn in synonyms:
                variants.append(re.sub(rf"\b{re.escape(term)}\b", syn, query, flags=re.IGNORECASE))
    # уникализируем, сохраняя порядок
    seen, out = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def extract_abbreviations(text: str, cfg: dict) -> list[str]:
    """Находит в тексте аббревиатуры из словаря."""
    found = []
    for abbr in (cfg.get("domain", {}).get("abbreviations") or {}):
        if re.search(rf"\b{re.escape(abbr)}\b", text):
            found.append(abbr)
    return found
