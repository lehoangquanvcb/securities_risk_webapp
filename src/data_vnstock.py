"""vnstock market data layer for the V3 CRO Command Center."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_RAW_DIR = Path("data/raw")


def _today() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def _ensure_dirs() -> None:
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_RAW_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_market_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Empty dataframe returned from vnstock")
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    rename_map = {
        "time": "date", "tradingDate": "date", "trading_date": "date", "date_time": "date", "Date": "date",
        "Close": "close", "closePrice": "close", "close_price": "close", "matchPrice": "close",
        "Open": "open", "High": "high", "Low": "low", "Volume": "volume",
        "value": "trading_value", "tradingValue": "trading_value", "trading_value": "trading_value",
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})
    if "date" not in data.columns:
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index().rename(columns={"index": "date"})
        else:
            raise ValueError(f"No date column in vnstock response for {symbol}: {data.columns.tolist()}")
    if "close" not in data.columns:
        raise ValueError(f"No close column in vnstock response for {symbol}: {data.columns.tolist()}")
    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "trading_value"] if c in data.columns]
    data = data[keep].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "trading_value"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    data["symbol"] = symbol.upper()
    if data.empty:
        raise ValueError(f"No valid price rows after normalizing vnstock data for {symbol}")
    return data


def sample_market_data(symbol: str = "VNINDEX", days: int = 520) -> pd.DataFrame:
    rng = pd.bdate_range(end=datetime.today(), periods=days)
    np.random.seed(abs(hash(symbol)) % (2**32))
    returns = np.random.normal(0.00025, 0.0115, len(rng))
    start_level = 1200 if symbol.upper().endswith("INDEX") or symbol.upper() in ["VN30", "HNXINDEX", "UPCOMINDEX"] else 25_000
    close = start_level * (1 + pd.Series(returns, index=rng)).cumprod()
    return pd.DataFrame({"date": rng, "close": close.round(2), "symbol": symbol.upper(), "data_source": "sample_offline"})


def fetch_market_data_vnstock(symbol: str = "VNINDEX", start: str = "2024-01-01", end: str | None = None, source: str = "VCI", interval: str = "1D") -> pd.DataFrame:
    end = end or _today()
    errors: list[str] = []
    try:
        from vnstock import Vnstock  # type: ignore
        stock = Vnstock().stock(symbol=symbol, source=source)
        df = stock.quote.history(start=start, end=end, interval=interval)
        out = _normalize_market_df(df, symbol)
        out["data_source"] = f"vnstock:{source}:Vnstock"
        return out
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Vnstock style failed: {exc}")
    try:
        from vnstock import Quote  # type: ignore
        quote = Quote(symbol=symbol, source=source)
        try:
            df = quote.history(start=start, end=end, interval=interval)
        except Exception:
            df = quote.history(start=start, end=end, interval="d")
        out = _normalize_market_df(df, symbol)
        out["data_source"] = f"vnstock:{source}:Quote"
        return out
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Quote style failed: {exc}")
    raise RuntimeError("; ".join(errors))


def load_market_data(symbol: str = "VNINDEX", start: str = "2024-01-01", end: str | None = None, source: str = "VCI", use_cache: bool = True, allow_sample_fallback: bool = True) -> tuple[pd.DataFrame, str]:
    _ensure_dirs()
    end = end or _today()
    cache_path = DEFAULT_CACHE_DIR / f"market_{symbol.upper()}_{source.upper()}_{start}_{end}.csv".replace(":", "-")
    try:
        df = fetch_market_data_vnstock(symbol=symbol, start=start, end=end, source=source)
        if use_cache:
            df.to_csv(cache_path, index=False)
        return df, f"LIVE vnstock data loaded: {symbol.upper()} from {source.upper()} ({len(df):,} rows)."
    except Exception as exc:  # noqa: BLE001
        if use_cache:
            existing = sorted(DEFAULT_CACHE_DIR.glob(f"market_{symbol.upper()}_{source.upper()}_*.csv"), reverse=True)
            if existing:
                df = pd.read_csv(existing[0])
                df = _normalize_market_df(df, symbol)
                df["data_source"] = "local_cache"
                return df, f"vnstock unavailable; using cached file {existing[0].name}. Error: {exc}"
        raw_path = DEFAULT_RAW_DIR / f"market_{symbol.upper()}.csv"
        if raw_path.exists():
            df = pd.read_csv(raw_path)
            df = _normalize_market_df(df, symbol)
            df["data_source"] = "local_raw_csv"
            return df, f"vnstock unavailable; using local raw CSV {raw_path}. Error: {exc}"
        if allow_sample_fallback:
            return sample_market_data(symbol=symbol), f"vnstock unavailable; using SAMPLE data only. Error: {exc}"
        raise


def load_multiple_market_data(symbols: Iterable[str], start: str = "2024-01-01", end: str | None = None, source: str = "VCI") -> tuple[pd.DataFrame, list[str]]:
    frames, messages = [], []
    for sym in symbols:
        sym = sym.strip().upper()
        if not sym:
            continue
        df, msg = load_market_data(symbol=sym, start=start, end=end, source=source)
        frames.append(df)
        messages.append(msg)
    return (pd.concat(frames, ignore_index=True), messages) if frames else (pd.DataFrame(), messages)
