"""vnstock market data layer for the V3 CRO Command Center.

V4 Final - Streamlit Cloud Stable

Design goals:
- Do NOT hang Streamlit Cloud when vnstock is rate-limited.
- Use cache first, then live vnstock only when app.py explicitly calls load_market_data().
- Use only the new vnstock 4 API path: from vnstock.api.quote import Quote.
- Avoid old deprecated APIs: from vnstock import Vnstock / Quote.
- Use VCI only to reduce API calls.
- Fall back safely to cache, raw CSV, then sample data.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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


def _today() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def _ensure_dirs() -> None:
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_RAW_DIR.mkdir(parents=True, exist_ok=True)


def normalize_source(source: str | None) -> str:
    if not source:
        return "VCI"
    return SOURCE_MAP.get(str(source).strip().upper(), "VCI")


def _normalize_interval(interval: str) -> str:
    interval = str(interval or "1D").strip()
    if interval.lower() in {"d", "day", "daily", "1d"}:
        return "1D"
    return interval


def _normalize_market_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Empty dataframe returned from vnstock")

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
    """Generate offline sample data for safe startup.

    This is marked with data_source='sample_offline' so users can see it is not live.
    """
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


def _latest_cache_for_symbol(symbol: str) -> Path | None:
    _ensure_dirs()
    files = sorted(DEFAULT_CACHE_DIR.glob(f"market_{symbol.upper()}_*.csv"), reverse=True)
    return files[0] if files else None


def _load_cache(symbol: str) -> tuple[pd.DataFrame, str] | None:
    cache_path = _latest_cache_for_symbol(symbol)
    if not cache_path:
        return None
    df = pd.read_csv(cache_path)
    df = _normalize_market_df(df, symbol)
    df["data_source"] = "local_cache"
    return df, f"Using cached market data: {cache_path.name}."


def _load_raw_csv(symbol: str) -> tuple[pd.DataFrame, str] | None:
    raw_path = DEFAULT_RAW_DIR / f"market_{symbol.upper()}.csv"
    if not raw_path.exists():
        return None
    df = pd.read_csv(raw_path)
    df = _normalize_market_df(df, symbol)
    df["data_source"] = "local_raw_csv"
    return df, f"Using local raw CSV: {raw_path}."


def _history_call(obj, start: str, end: str, interval: str) -> pd.DataFrame:
    interval = _normalize_interval(interval)
    tries = [
        {"start": start, "end": end, "interval": interval},
        {"start": start, "end": end, "interval": interval.lower()},
        {"start": start, "end": end, "interval": "1D"},
        {"start": start, "end": end, "interval": "d"},
        {"start": start, "end": end},
    ]
    errors = []
    for kwargs in tries:
        try:
            return obj.history(**kwargs)
        except TypeError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("; ".join(errors))


def _fetch_live_vnstock_api_quote(symbol: str, start: str, end: str, source: str, interval: str) -> pd.DataFrame:
    """Preferred vnstock 4 API. Avoids deprecated `from vnstock import Vnstock`."""
    from vnstock.api.quote import Quote  # type: ignore

    source = normalize_source(source)
    quote = Quote(symbol=symbol, source=source)
    df = _history_call(quote, start=start, end=end, interval=interval)
    out = _normalize_market_df(df, symbol)
    out["data_source"] = f"vnstock4_api:{source}:Quote"
    return out


def fetch_market_data_vnstock(
    symbol: str = "VNINDEX",
    start: str = "2024-01-01",
    end: str | None = None,
    source: str = "VCI",
    interval: str = "1D",
    timeout_seconds: int = 12,
) -> pd.DataFrame:
    """Fetch one symbol from vnstock with a hard timeout."""
    end = end or _today()
    source = normalize_source(source)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fetch_live_vnstock_api_quote, symbol.upper(), start, end, source, interval)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            raise TimeoutError(f"vnstock live request timed out after {timeout_seconds} seconds") from exc


def load_market_data(
    symbol: str = "VNINDEX",
    start: str = "2024-01-01",
    end: str | None = None,
    source: str = "VCI",
    use_cache: bool = True,
    allow_sample_fallback: bool = True,
    prefer_cache: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Load market data safely: live -> cache -> raw CSV -> sample."""
    _ensure_dirs()
    end = end or _today()
    symbol = symbol.strip().upper()
    source = normalize_source(source)

    if prefer_cache and use_cache:
        cached = _load_cache(symbol)
        if cached is not None:
            return cached

    try:
        df = fetch_market_data_vnstock(symbol=symbol, start=start, end=end, source=source, timeout_seconds=12)
        cache_path = DEFAULT_CACHE_DIR / f"market_{symbol}_{source}_{start}_{end}.csv".replace(":", "-")
        if use_cache:
            df.to_csv(cache_path, index=False)
        return df, f"LIVE vnstock data loaded: {symbol} from {source} ({len(df):,} rows)."
    except Exception as exc:  # noqa: BLE001
        if use_cache:
            cached = _load_cache(symbol)
            if cached is not None:
                df, msg = cached
                return df, f"vnstock unavailable; {msg} Error: {exc}"
        raw = _load_raw_csv(symbol)
        if raw is not None:
            df, msg = raw
            return df, f"vnstock unavailable; {msg} Error: {exc}"
        if allow_sample_fallback:
            df = sample_market_data(symbol=symbol)
            return df, f"vnstock unavailable; using SAMPLE data only. Error: {exc}"
        raise


def load_multiple_market_data(
    symbols: Iterable[str],
    start: str = "2024-01-01",
    end: str | None = None,
    source: str = "VCI",
) -> tuple[pd.DataFrame, list[str]]:
    """Load watchlist safely without live requests to avoid rate limits."""
    frames, messages = [], []
    seen: set[str] = set()
    clean_symbols: list[str] = []
    for sym in symbols:
        sym = sym.strip().upper()
        if sym and sym not in seen:
            clean_symbols.append(sym)
            seen.add(sym)
    for sym in clean_symbols:
        cached = _load_cache(sym)
        if cached is not None:
            df, msg = cached
        else:
            raw = _load_raw_csv(sym)
            if raw is not None:
                df, msg = raw
            else:
                df = sample_market_data(sym)
                msg = f"Watchlist safe mode: using SAMPLE data for {sym}; no live vnstock call to avoid rate limits."
        frames.append(df)
        messages.append(msg)
    return (pd.concat(frames, ignore_index=True), messages) if frames else (pd.DataFrame(), messages)
