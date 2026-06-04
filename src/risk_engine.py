from __future__ import annotations

import numpy as np
import pandas as pd


def status_high_bad(value, amber, red):
    if pd.isna(value):
        return "N/A"
    if value >= red:
        return "Red"
    if value >= amber:
        return "Amber"
    return "Green"


def status_low_bad(value, amber, red):
    if pd.isna(value):
        return "N/A"
    if value <= red:
        return "Red"
    if value <= amber:
        return "Amber"
    return "Green"


def compute_market_metrics(price_df: pd.DataFrame, portfolio_value_vnd: float = 100_000_000_000) -> dict:
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["drawdown_20d"] = df["close"] / df["close"].rolling(20, min_periods=2).max() - 1
    df["drawdown_from_peak"] = df["close"] / df["close"].cummax() - 1
    df["vol_20d"] = df["return_1d"].rolling(20, min_periods=5).std() * np.sqrt(252)
    df["ma_20"] = df["close"].rolling(20, min_periods=5).mean()
    df["ma_50"] = df["close"].rolling(50, min_periods=10).mean()
    df["var_95_1d"] = df["return_1d"].rolling(252, min_periods=60).quantile(0.05)
    df["var_99_1d"] = df["return_1d"].rolling(252, min_periods=60).quantile(0.01)
    rolling_mean = df["return_1d"].rolling(252, min_periods=60).mean()
    rolling_std = df["return_1d"].rolling(252, min_periods=60).std()
    df["parametric_var_95_1d"] = rolling_mean - 1.645 * rolling_std
    df["parametric_var_99_1d"] = rolling_mean - 2.326 * rolling_std

    def es(x, q):
        s = pd.Series(x).dropna()
        if len(s) < 60:
            return np.nan
        threshold = s.quantile(q)
        return s[s <= threshold].mean()

    df["expected_shortfall_95_1d"] = df["return_1d"].rolling(252, min_periods=60).apply(lambda x: es(x, 0.05), raw=False)
    df["expected_shortfall_99_1d"] = df["return_1d"].rolling(252, min_periods=60).apply(lambda x: es(x, 0.01), raw=False)
    latest = df.iloc[-1]
    def f(col):
        return float(latest.get(col, np.nan))
    return {
        "latest_close": f("close"),
        "return_1d": f("return_1d"),
        "return_5d": f("return_5d"),
        "drawdown_20d": f("drawdown_20d"),
        "drawdown_from_peak": f("drawdown_from_peak"),
        "vol_20d": f("vol_20d"),
        "var_95_1d": f("var_95_1d"),
        "var_99_1d": f("var_99_1d"),
        "parametric_var_95_1d": f("parametric_var_95_1d"),
        "parametric_var_99_1d": f("parametric_var_99_1d"),
        "expected_shortfall_95_1d": f("expected_shortfall_95_1d"),
        "expected_shortfall_99_1d": f("expected_shortfall_99_1d"),
        "var_95_vnd": abs(f("var_95_1d")) * portfolio_value_vnd,
        "es_95_vnd": abs(f("expected_shortfall_95_1d")) * portfolio_value_vnd,
        "data": df,
    }


def compute_margin_metrics(margin_df: pd.DataFrame, total_margin_limit: float) -> dict:
    df = margin_df.copy()
    for col in ["exposure_vnd", "loan_vnd", "collateral_value_vnd", "ltv", "margin_call_trigger", "force_sell_trigger"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "exposure_vnd" not in df.columns and "loan_vnd" in df.columns:
        df["exposure_vnd"] = df["loan_vnd"]
    exposure = float(df.get("exposure_vnd", pd.Series(dtype=float)).sum())
    top10 = float(df.sort_values("exposure_vnd", ascending=False).head(10)["exposure_vnd"].sum()) if "exposure_vnd" in df else 0
    top1 = float(df.sort_values("exposure_vnd", ascending=False).head(1)["exposure_vnd"].sum()) if "exposure_vnd" in df else 0
    mc_trigger = df.get("margin_call_trigger", pd.Series(0.60, index=df.index))
    fs_trigger = df.get("force_sell_trigger", pd.Series(0.70, index=df.index))
    return {
        "total_exposure": exposure,
        "margin_utilization": exposure / total_margin_limit if total_margin_limit else np.nan,
        "top1_concentration": top1 / exposure if exposure else np.nan,
        "top10_concentration": top10 / exposure if exposure else np.nan,
        "num_margin_call": int((df.get("ltv", pd.Series(dtype=float)) >= mc_trigger).sum()),
        "num_force_sell": int((df.get("ltv", pd.Series(dtype=float)) >= fs_trigger).sum()),
        "data": df,
    }


def compute_public_kri(public_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if public_df is None or public_df.empty:
        return pd.DataFrame(columns=["risk_area", "indicator", "current", "amber", "red", "status", "source"])
    for _, r in public_df.iterrows():
        if pd.isna(r.get("amber")) or pd.isna(r.get("red")):
            continue
        direction = str(r.get("direction", "high_bad")).lower()
        val, amber, red = r.get("value"), r.get("amber"), r.get("red")
        status = status_low_bad(val, amber, red) if direction == "low_bad" else status_high_bad(val, amber, red)
        rows.append([r.get("risk_area"), r.get("indicator"), val, amber, red, status, r.get("source")])
    return pd.DataFrame(rows, columns=["risk_area", "indicator", "current", "amber", "red", "status", "source"])


def build_kri_table(market: dict, margin: dict, liquidity: dict, operational: dict, public_df: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = [
        ["Market risk", "1D market return", market["return_1d"], -0.03, -0.05, status_low_bad(market["return_1d"], -0.03, -0.05), "vnstock"],
        ["Market risk", "VNINDEX 20D drawdown", market["drawdown_20d"], -0.07, -0.12, status_low_bad(market["drawdown_20d"], -0.07, -0.12), "vnstock"],
        ["Market risk", "20D annualized volatility", market["vol_20d"], 0.18, 0.25, status_high_bad(market["vol_20d"], 0.18, 0.25), "vnstock"],
        ["Market risk", "Historical VaR 95%", abs(market["var_95_1d"]), 0.025, 0.04, status_high_bad(abs(market["var_95_1d"]), 0.025, 0.04), "vnstock"],
        ["Liquidity risk", "Liquidity buffer ratio", liquidity.get("liquidity_buffer_ratio"), 1.20, 1.00, status_low_bad(liquidity.get("liquidity_buffer_ratio"), 1.20, 1.00), "V3 workbook: Liquidity_Risk"],
        ["Margin risk", "Margin utilization", margin["margin_utilization"], 0.75, 0.90, status_high_bad(margin["margin_utilization"], 0.75, 0.90), "V3 workbook: Margin_Book"],
        ["Margin risk", "Number of margin call accounts", margin["num_margin_call"], 5, 10, status_high_bad(margin["num_margin_call"], 5, 10), "V3 workbook: Margin_Book"],
        ["Concentration", "Top 1 client concentration", margin["top1_concentration"], 0.10, 0.15, status_high_bad(margin["top1_concentration"], 0.10, 0.15), "V3 workbook: Margin_Book"],
        ["Concentration", "Top 10 client concentration", margin["top10_concentration"], 0.30, 0.40, status_high_bad(margin["top10_concentration"], 0.30, 0.40), "V3 workbook: Margin_Book"],
        ["Operational risk", "Monthly incident count", operational.get("monthly_incident_count"), 20, 30, status_high_bad(operational.get("monthly_incident_count"), 20, 30), "V3 workbook: Operational_Risk"],
        ["Operational risk", "High severity cases", operational.get("high_severity_cases"), 3, 5, status_high_bad(operational.get("high_severity_cases"), 3, 5), "V3 workbook: Operational_Risk"],
        ["Compliance", "Open / overdue exceptions", operational.get("overdue_exceptions"), 2, 5, status_high_bad(operational.get("overdue_exceptions"), 2, 5), "V3 workbook: Operational_Risk"],
    ]
    base = pd.DataFrame(rows, columns=["risk_area", "indicator", "current", "amber", "red", "status", "source"])
    pub = compute_public_kri(public_df) if public_df is not None else pd.DataFrame()
    if not pub.empty:
        base = pd.concat([base, pub], ignore_index=True)
    return base


def stress_test(total_exposure: float, prop_trading_value: float = 100_000_000_000, scenarios: pd.DataFrame | None = None) -> pd.DataFrame:
    if scenarios is None or scenarios.empty or "Scenario" not in scenarios.columns:
        scenarios = pd.DataFrame({
            "Scenario": ["Base", "Moderate", "Severe", "Liquidity freeze", "2008-style shock"],
            "VNINDEX Shock": [-0.05, -0.12, -0.25, -0.18, -0.35],
            "Liquidity Haircut": [0.00, -0.10, -0.25, -0.40, -0.50],
            "Margin Loan Shock": [0.05, 0.12, 0.25, 0.20, 0.35],
        })
    out = scenarios.copy()
    out.columns = [str(c).strip() for c in out.columns]
    rename = {
        "Scenario": "scenario",
        "VNINDEX Shock": "vnindex_shock",
        "Liquidity Haircut": "liquidity_haircut",
        "Margin Loan Shock": "margin_loan_shock",
        "Foreign Flow Shock (VND bn)": "foreign_flow_shock_bn",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    for c in ["vnindex_shock", "liquidity_haircut", "margin_loan_shock", "foreign_flow_shock_bn"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    if "margin_loan_shock" not in out.columns:
        out["margin_loan_shock"] = out.get("collateral_haircut", 0.15)
    out["estimated_market_loss_vnd"] = abs(prop_trading_value * out.get("vnindex_shock", 0))
    out["estimated_margin_loss_vnd"] = total_exposure * abs(out.get("margin_loan_shock", 0)) * 0.35
    out["estimated_liquidity_gap_vnd"] = abs(out.get("liquidity_haircut", 0)) * (total_exposure + prop_trading_value) * 0.20
    out["total_estimated_loss_vnd"] = out["estimated_market_loss_vnd"] + out["estimated_margin_loss_vnd"] + out["estimated_liquidity_gap_vnd"]
    return out


def overall_status(kri: pd.DataFrame) -> str:
    if kri.empty:
        return "N/A"
    if (kri["status"] == "Red").any():
        return "Red"
    if (kri["status"] == "Amber").any():
        return "Amber"
    return "Green"
