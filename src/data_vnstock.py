"""vnstock market data layer for the V3 CRO Command Center.

Version: V4-compatible fallback layer

What this file does:
- Works with vnstock 4.x when available.
- Keeps backward compatibility with vnstock 3.x style calls.
- Uses only commonly supported public sources: VCI, TCBS, MSN.
- Automatically tries several API styles and sources before using cache/raw/sample.
- Produces clear messages so the Streamlit UI shows whether data is live, cache, raw CSV, or sample.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_RAW_DIR = Path("data/raw")

SUPPORTED_SOURCES = ["VCI", "TCBS", "MSN"]
SOURCE_MAP = {
    "VCI": "VCI",
    "TCBS": "TCBS",
    "MSN": "MSN",
    "KBS": "VCI",
    "FMP": "VCI",
    "SSI": "VCI",
    "DNSE": "VCI",
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


def source_candidates(source: str | None) -> list[str]:
    first = normalize_source(source)
    candidates = [first] + [s for s in SUPPORTED_SOURCES if s != first]
    return list(dict.fromkeys(candidates))


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
        "time": "date",
        "tradingDate": "date",
        "trading_date": "date",
        "date_time": "date",
        "datetime": "date",
        "Date": "date",
        "date": "date",
        "Close": "close",
        "close": "close",
        "closePrice": "close",
        "close_price": "close",
        "matchPrice": "close",
        "Open": "open",
        "open": "open",
        "openPrice": "open",
        "High": "high",
        "high": "high",
        "highPrice": "high",
        "Low": "low",
        "low": "low",
        "lowPrice": "low",
        "Volume": "volume",
        "volume": "volume",
        "value": "trading_value",
        "tradingValue": "trading_value",
        "trading_value": "trading_value",
        "matchedValue": "trading_value",
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
    return pd.DataFrame(
        {
            "date": rng,
            "close": close.round(2),
            "symbol": symbol.upper(),
            "data_source": "sample_offline",
        }
    )


def _history_call(obj, start: str, end: str, interval: str) -> pd.DataFrame:
    """Call history with a few interval formats used by different vnstock versions."""
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


def _fetch_vnstock_v4_direct_quote(symbol: str, start: str, end: str, source: str, interval: str) -> pd.DataFrame:
    """Use vnstock_data direct explorer classes introduced around vnstock 4.x."""
    source = normalize_source(source)

    if source == "VCI":
        from vnstock_data.explorer.vci.quote import Quote  # type: ignore
        quote = Quote(symbol=symbol)
    elif source == "TCBS":
        from vnstock_data.explorer.tcbs.quote import Quote  # type: ignore
        quote = Quote(symbol=symbol)
    elif source == "MSN":
        from vnstock_data.explorer.msn.quote import Quote  # type: ignore
        quote = Quote(symbol=symbol)
    else:
        raise ValueError(f"Unsupported direct vnstock_data source: {source}")

    df = _history_call(quote, start=start, end=end, interval=interval)
    out = _normalize_market_df(df, symbol)
    out["data_source"] = f"vnstock4:{source}:direct_quote"
    return out


def _fetch_vnstock_v4_facade(symbol: str, start: str, end: str, source: str, interval: str) -> pd.DataFrame:
    """Use unified Vnstock facade. This also works for some vnstock 3.x installs."""
    from vnstock import Vnstock  # type: ignore

    source = normalize_source(source)
    stock = Vnstock().stock(symbol=symbol, source=source)
    df = _history_call(stock.quote, start=start, end=end, interval=interval)
    out = _normalize_market_df(df, symbol)
    out["data_source"] = f"vnstock4:{source}:facade"
    return out


def _fetch_vnstock_legacy_quote(symbol: str, start: str, end: str, source: str, interval: str) -> pd.DataFrame:
    """Fallback for older vnstock 3.x Quote class."""
    from vnstock import Quote  # type: ignore

    source = normalize_source(source)
    quote = Quote(symbol=symbol, source=source)
    df = _history_call(quote, start=start, end=end, interval=interval)
    out = _normalize_market_df(df, symbol)
    out["data_source"] = f"vnstock3:{source}:Quote"
    return out


def fetch_market_data_vnstock(
    symbol: str = "VNINDEX",
    start: str = "2024-01-01",
    end: str | None = None,
    source: str = "VCI",
    interval: str = "1D",
) -> pd.DataFrame:
    """Fetch one symbol from vnstock using one explicit source."""
    end = end or _today()
    source = normalize_source(source)
    errors: list[str] = []

    fetchers = [
        ("vnstock4_direct_quote", _fetch_vnstock_v4_direct_quote),
        ("vnstock4_facade", _fetch_vnstock_v4_facade),
        ("vnstock_legacy_quote", _fetch_vnstock_legacy_quote),
    ]

    for name, fetcher in fetchers:
        try:
            return fetcher(symbol=symbol, start=start, end=end, source=source, interval=interval)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source}/{name} failed: {exc}")

    raise RuntimeError("; ".join(errors))


def fetch_market_data_vnstock_auto(
    symbol: str = "VNINDEX",
    start: str = "2024-01-01",
    end: str | None = None,
    source: str = "VCI",
    interval: str = "1D",
) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []

    for src in source_candidates(source):
        try:
            df = fetch_market_data_vnstock(
                symbol=symbol,
                start=start,
                end=end,
                source=src,
                interval=interval,
            )
            return df, src
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{src}: {exc}")

    raise RuntimeError("All vnstock sources failed: " + " | ".join(errors))


def load_market_data(
    symbol: str = "VNINDEX",
    start: str = "2024-01-01",
    end: str | None = None,
    source: str = "VCI",
    use_cache: bool = True,
    allow_sample_fallback: bool = True,
) -> tuple[pd.DataFrame, str]:
    _ensure_dirs()
    end = end or _today()

    requested_source = str(source or "VCI").upper()

    try:
        df, used_source = fetch_market_data_vnstock_auto(
            symbol=symbol,
            start=start,
            end=end,
            source=requested_source,
        )

        cache_path = DEFAULT_CACHE_DIR / f"market_{symbol.upper()}_{used_source.upper()}_{start}_{end}.csv".replace(":", "-")
        if use_cache:
            df.to_csv(cache_path, index=False)

        if requested_source != used_source:
            msg = (
                f"LIVE vnstock data loaded: {symbol.upper()} from {used_source.upper()} "
                f"({len(df):,} rows). Requested source {requested_source} was mapped/fallbacked."
            )
        else:
            msg = f"LIVE vnstock data loaded: {symbol.upper()} from {used_source.upper()} ({len(df):,} rows)."
        return df, msg

    except Exception as exc:  # noqa: BLE001
        if use_cache:
            existing = sorted(DEFAULT_CACHE_DIR.glob(f"market_{symbol.upper()}_*.csv"), reverse=True)
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


def load_multiple_market_data(
    symbols: Iterable[str],
    start: str = "2024-01-01",
    end: str | None = None,
    source: str = "VCI",
) -> tuple[pd.DataFrame, list[str]]:
    frames, messages = [], []

    for sym in symbols:
        sym = sym.strip().upper()
        if not sym:
            continue
        df, msg = load_market_data(symbol=sym, start=start, end=end, source=source)
        frames.append(df)
        messages.append(msg)

    return (pd.concat(frames, ignore_index=True), messages) if frames else (pd.DataFrame(), messages)
