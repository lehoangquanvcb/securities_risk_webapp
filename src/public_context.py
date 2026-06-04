from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.excel_io import read_public_context

PUBLIC_CONTEXT_PATH = Path("data/raw/public_macro_context.csv")


def load_public_context(path: str | Path = PUBLIC_CONTEXT_PATH, workbook_path: str | Path | None = None) -> pd.DataFrame:
    # Prefer the V3 workbook's Public_Context sheet, fall back to CSV.
    try:
        df = read_public_context(workbook_path)
        if not df.empty:
            return df
    except Exception:
        pass
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["as_of_date", "indicator", "value", "unit", "risk_area", "amber", "red", "direction", "source", "note"])
    df = pd.read_csv(p)
    for col in ["value", "amber", "red"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def latest_public_context(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "as_of_date" in out.columns:
        out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
        out = out.sort_values(["indicator", "as_of_date"]).groupby("indicator", as_index=False).tail(1)
    return out.sort_values([c for c in ["risk_area", "indicator"] if c in out.columns])
