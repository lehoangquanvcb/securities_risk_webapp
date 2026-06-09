"""Update real market data locally, then commit CSV to GitHub.

Usage:
    python update_market_data.py

After running:
    git add data/raw/*.csv
    git commit -m "Update market data"
    git push origin main
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import time
import pandas as pd

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SYMBOLS = ["VNINDEX", "HNXINDEX", "UPCOMINDEX", "VN30", "VCB", "MBB", "FPT", "HPG", "MWG", "VHM", "SSI", "VND"]
DEFAULT_START = "2024-01-01"
DEFAULT_END = date.today().strftime("%Y-%m-%d")


def _normalize_market_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
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
    data["data_source"] = "vnstock_local_update"
    return data


def _history_call(obj, start: str, end: str, interval: str = "1D") -> pd.DataFrame:
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
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("; ".join(errors))


def fetch_symbol(symbol: str, start: str, end: str, source: str = "VCI") -> pd.DataFrame:
    symbol = symbol.upper()
    try:
        from vnstock.api.quote import Quote  # type: ignore
        q = Quote(symbol=symbol, source=source)
        df = _history_call(q, start=start, end=end, interval="1D")
        return _normalize_market_df(df, symbol)
    except Exception as exc1:
        print(f"[WARN] vnstock.api.quote failed for {symbol}: {exc1}")

    try:
        from vnstock_data.explorer.vci.quote import Quote  # type: ignore
        q = Quote(symbol=symbol)
        df = _history_call(q, start=start, end=end, interval="1D")
        return _normalize_market_df(df, symbol)
    except Exception as exc2:
        print(f"[WARN] vnstock_data explorer failed for {symbol}: {exc2}")

    raise RuntimeError(f"Cannot fetch {symbol} from vnstock.")


def main() -> None:
    print(f"Updating market data: {DEFAULT_SYMBOLS}")
    print(f"Date range: {DEFAULT_START} -> {DEFAULT_END}")
    for symbol in DEFAULT_SYMBOLS:
        try:
            df = fetch_symbol(symbol, DEFAULT_START, DEFAULT_END, source="VCI")
            out = RAW_DIR / f"market_{symbol}.csv"
            df.to_csv(out, index=False)
            print(f"[OK] {symbol}: {len(df):,} rows -> {out}")
            time.sleep(3)
        except Exception as exc:
            print(f"[ERROR] {symbol}: {exc}")
    sector_map = pd.DataFrame([
        {"symbol": "VCB", "sector": "Banking"},
        {"symbol": "MBB", "sector": "Banking"},
        {"symbol": "FPT", "sector": "Technology"},
        {"symbol": "HPG", "sector": "Steel"},
        {"symbol": "MWG", "sector": "Retail"},
        {"symbol": "VHM", "sector": "Real Estate"},
        {"symbol": "SSI", "sector": "Securities"},
        {"symbol": "VND", "sector": "Securities"},
        {"symbol": "VNINDEX", "sector": "Market Index"},
        {"symbol": "VN30", "sector": "Market Index"},
        {"symbol": "HNXINDEX", "sector": "Market Index"},
        {"symbol": "UPCOMINDEX", "sector": "Market Index"},
    ])
    sector_map.to_csv(RAW_DIR / "sector_map.csv", index=False)
    print(f"[OK] sector_map.csv -> {RAW_DIR / 'sector_map.csv'}")
    print("Done. Commit generated CSV files to GitHub.")


if __name__ == "__main__":
    main()
