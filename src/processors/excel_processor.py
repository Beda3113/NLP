"""Обработчик Excel/CSV."""
from pathlib import Path

import pandas as pd

from .base import BaseProcessor


class ExcelProcessor(BaseProcessor):
    def __init__(self, config: dict):
        self.config = config

    def supported_extensions(self) -> list[str]:
        return ["xlsx", "xls", "csv"]

    def extract(self, file_path: Path) -> str:
        if file_path.name.startswith("~$"):
            return ""

        ext = file_path.suffix.lower().lstrip(".")
        parts: list[str] = []
        try:
            if ext == "csv":
                df = pd.read_csv(file_path)
                parts.append(self._df_to_text(df, "CSV"))
            elif ext == "xls":
                df = self._read_any_excel(file_path, engines=["calamine", "xlrd"])
                if df is not None:
                    parts.append(self._df_to_text(df, "Sheet"))
            else:  # xlsx
                xls = pd.ExcelFile(file_path, engine="openpyxl")
                for sheet in xls.sheet_names:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl")
                        parts.append(self._df_to_text(df, sheet))
                    except Exception:
                        continue
        except Exception:
            return ""
        return "\n\n".join(parts)

    @staticmethod
    def _read_any_excel(file_path: Path, engines: list[str]):
        for eng in engines:
            try:
                return pd.read_excel(file_path, engine=eng)
            except Exception:
                continue
        return None

    def _df_to_text(self, df: pd.DataFrame, sheet_name: str) -> str:
        lines = [f"Лист: {sheet_name}"]
        for _, row in df.iterrows():
            cells = [f"{col}={val}" for col, val in row.items() if pd.notna(val)]
            if cells:
                lines.append(" | ".join(cells))
        return "\n".join(lines)
