"""Market data layer for V4 Enterprise CRO Command Center.

Streamlit-safe design:
- Streamlit does NOT call vnstock directly.
- Streamlit reads real market data from data/raw/market_<SYMBOL>.csv first.
- If raw CSV is missing, it reads data/cache.
- If both are missing, it uses sample data and clearly labels it.

To update real data:
    python update_market_data.py
then commit/push the generated CSV files under data/raw/.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_RAW_DIR = Path("data/raw")

SUPPORTED_SOURCES = ["VCI"]
SOURCE_MAP = {
    "VCI": "VCI",
    "TCBS": "VCI",
    "MSN": "VCI",
    "KBS": "VCI",
    "FMP": "VCI",
    "SSI": "VCI",
    "DNSE": "VCI",
    "MAS": "VCI",
}


def _ensure_dirs() -> None:
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_RAW_DIR.mkdir(parents=True, exist_ok=True)


def normalize_source(source: str | None) -> str:
    if not source:
        return "VCI"
    return SOURCE_MAP.get(str(source).strip().upper(), "VCI")


def _normalize_market_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Empty market dataframe")

    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    rename_map = {
        "time": "date", "tradingDate": "date", "trading_date": "date",
        "date_time": "date", "datetime": "date", "Date": "date", "date": "date",
        "Close": "close", "close": "close", "closePrice": "close",
        "close_price": "close", "matchPrice": "close",
        "Open": "open", "open": "open", "openPrice": "open",
        "High": "high", "high": "high", "highPrice": "high",
        "Low": "low", "low": "low", "lowPrice": "low",
        "Volume": "volume", "volume": "volume",
        "value": "trading_value", "tradingValue": "trading_value",
        "trading_value": "trading_value", "matchedValue": "trading_value",
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

    if "date" not in data.columns:
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index().rename(columns={"index": "date"})
        else:
            raise ValueError(f"No date column for {symbol}: {data.columns.tolist()}")

    if "close" not in data.columns:
        raise ValueError(f"No close column for {symbol}: {data.columns.tolist()}")

    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "trading_value"] if c in data.columns]
    data = data[keep].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    for c in ["open", "high", "low", "close", "volume", "trading_value"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    data = data.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    data["symbol"] = symbol.upper()

    if data.empty:
        raise ValueError(f"No valid market rows for {symbol}")

    return data


def sample_market_data(symbol: str = "VNINDEX", days: int = 520) -> pd.DataFrame:
    rng = pd.bdate_range(end=datetime.today(), periods=days)
    np.random.seed(abs(hash(symbol)) % (2**32))
    returns = np.random.normal(0.00025, 0.0115, len(rng))
    start_level = 1200 if symbol.upper().endswith("INDEX") or symbol.upper() in ["VN30", "HNXINDEX", "UPCOMINDEX"] else 25_000
    close = start_level * (1 + pd.Series(returns, index=rng)).cumprod()
    return pd.DataFrame({
        "date": rng,
        "close": close.round(2),
        "symbol": symbol.upper(),
        "data_source": "sample_offline",
    })


def _raw_paths(symbol: str) -> list[Path]:
    symbol = symbol.upper()
    return [
        DEFAULT_RAW_DIR / f"market_{symbol}.csv",
        DEFAULT_RAW_DIR / f"{symbol}.csv",
        DEFAULT_RAW_DIR / f"market_{symbol.lower()}.csv",
    ]


def _latest_cache_for_symbol(symbol: str) -> Path | None:
    _ensure_dirs()
    files = sorted(DEFAULT_CACHE_DIR.glob(f"market_{symbol.upper()}_*.csv"), reverse=True)
    return files[0] if files else None


def _load_csv(path: Path, symbol: str, source_label: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    df = _normalize_market_df(df, symbol)
    df["data_source"] = source_label
    return df, f"Using {source_label}: {path} ({len(df):,} rows)."


def _load_raw_csv(symbol: str) -> tuple[pd.DataFrame, str] | None:
    _ensure_dirs()
    for path in _raw_paths(symbol):
        if path.exists():
            return _load_csv(path, symbol, "local_raw_csv")
    return None


def _load_cache(symbol: str) -> tuple[pd.DataFrame, str] | None:
    path = _latest_cache_for_symbol(symbol)
    if path:
        return _load_csv(path, symbol, "local_cache")
    return None


def load_market_data(
    symbol: str = "VNINDEX",
    start: str = "2024-01-01",
    end: str | None = None,
    source: str = "VCI",
    use_cache: bool = True,
    allow_sample_fallback: bool = True,
    prefer_cache: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Load market data without any live vnstock request."""
    _ensure_dirs()
    symbol = symbol.strip().upper()

    raw = _load_raw_csv(symbol)
    if raw is not None:
        return raw

    if use_cache:
        cached = _load_cache(symbol)
        if cached is not None:
            return cached

    if allow_sample_fallback:
        df = sample_market_data(symbol=symbol)
        return df, f"No local market CSV/cache found for {symbol}; using SAMPLE data only."

    raise FileNotFoundError(
        f"No local market data found for {symbol}. Run update_market_data.py locally, then commit data/raw/market_{symbol}.csv."
    )


def load_multiple_market_data(
    symbols: Iterable[str],
    start: str = "2024-01-01",
    end: str | None = None,
    source: str = "VCI",
) -> tuple[pd.DataFrame, list[str]]:
    frames, messages = [], []
    seen: set[str] = set()
    clean_symbols: list[str] = []
    for sym in symbols:
        sym = sym.strip().upper()
        if sym and sym not in seen:
            clean_symbols.append(sym)
            seen.add(sym)

    for sym in clean_symbols:
        df, msg = load_market_data(symbol=sym, start=start, end=end, source=source)
        frames.append(df)
        messages.append(msg)

    return (pd.concat(frames, ignore_index=True), messages) if frames else (pd.DataFrame(), messages)
