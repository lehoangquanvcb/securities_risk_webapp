from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

MASTER_WORKBOOK = Path("data/securities_risk_management_v3_master.xlsx")


def find_master_workbook(path: str | Path | None = None) -> Path:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates += [
        MASTER_WORKBOOK,
        Path("securities_risk_management_v3_master.xlsx"),
        Path("data/securities_risk_management_master.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find V3 master workbook. Put securities_risk_management_v3_master.xlsx in the data/ folder."
    )


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop completely empty rows/columns and Excel unnamed blank columns.
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df.reset_index(drop=True)


def read_v3_sheet(sheet_name: str, workbook_path: str | Path | None = None, header: int = 2) -> pd.DataFrame:
    p = find_master_workbook(workbook_path)
    try:
        df = pd.read_excel(p, sheet_name=sheet_name, header=header)
        return _clean_df(df)
    except ValueError:
        return pd.DataFrame()


def read_public_context(workbook_path: str | Path | None = None) -> pd.DataFrame:
    df = read_v3_sheet("Public_Context", workbook_path)
    if df.empty:
        return df
    rename = {
        "Item": "indicator",
        "Value": "value",
        "Unit": "unit",
        "As of": "as_of_date",
        "Source": "source",
        "Notes": "note",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for c in ["value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "risk_area" not in df.columns:
        df["risk_area"] = df.get("indicator", "Public context").astype(str).map(_infer_risk_area)
    # Default thresholds for public context when V3 workbook only has values.
    defaults: dict[str, tuple[float, float, str]] = {
        "SBV refinancing rate": (5.0, 6.0, "high_bad"),
        "SBV discount rate": (4.0, 5.0, "high_bad"),
        "10Y Government bond yield": (3.5, 4.5, "high_bad"),
        "USD/VND": (26000, 27000, "high_bad"),
        "Foreign net flow": (-1500, -3000, "low_bad"),
    }
    df["amber"] = df.apply(lambda r: defaults.get(str(r.get("indicator")), (None, None, "high_bad"))[0], axis=1)
    df["red"] = df.apply(lambda r: defaults.get(str(r.get("indicator")), (None, None, "high_bad"))[1], axis=1)
    df["direction"] = df.apply(lambda r: defaults.get(str(r.get("indicator")), (None, None, "high_bad"))[2], axis=1)
    return df


def _infer_risk_area(name: str) -> str:
    text = name.lower()
    if "bond" in text or "yield" in text or "rate" in text:
        return "Interest rate risk"
    if "usd" in text or "fx" in text or "foreign" in text:
        return "Market risk"
    return "Public context"


def load_risk_limits(workbook_path: str | Path | None = None) -> pd.DataFrame:
    return read_v3_sheet("Risk_Limits", workbook_path)


def load_kri_library(workbook_path: str | Path | None = None) -> pd.DataFrame:
    return read_v3_sheet("KRI_Library", workbook_path)


def load_risk_appetite(workbook_path: str | Path | None = None) -> pd.DataFrame:
    return read_v3_sheet("Risk_Appetite", workbook_path)


def load_stress_template(workbook_path: str | Path | None = None) -> pd.DataFrame:
    return read_v3_sheet("Stress_Test", workbook_path)


def load_margin_book(workbook_path: str | Path | None = None) -> tuple[pd.DataFrame, str]:
    df = read_v3_sheet("Margin_Book", workbook_path)
    if df.empty:
        return pd.DataFrame(), "V3 workbook has no Margin_Book data."
    rename = {
        "Account": "client_id",
        "Customer Group": "customer_group",
        "Ticker": "ticker",
        "Sector": "sector",
        "Collateral Value (VND bn)": "collateral_value_bn",
        "Margin Loan (VND bn)": "loan_bn",
        "Equity (VND bn)": "equity_bn",
        "Margin Ratio": "ltv",
        "Maintenance Req.": "maintenance_requirement",
        "Excess / Shortfall": "excess_shortfall",
        "Status": "status",
        "Notes": "notes",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = ["client_id", "ticker", "sector", "collateral_value_bn", "loan_bn"]
    available = [c for c in required if c in df.columns]
    df = df.dropna(subset=available[:1]) if available else df
    for c in ["collateral_value_bn", "loan_bn", "equity_bn", "ltv", "maintenance_requirement", "excess_shortfall"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "collateral_value_bn" in df.columns:
        df["collateral_value_vnd"] = df["collateral_value_bn"] * 1_000_000_000
    if "loan_bn" in df.columns:
        df["loan_vnd"] = df["loan_bn"] * 1_000_000_000
        df["exposure_vnd"] = df["loan_vnd"]
    if "ltv" not in df.columns or df["ltv"].isna().all():
        df["ltv"] = df["loan_vnd"] / df["collateral_value_vnd"]
    if "margin_call_trigger" not in df.columns:
        df["margin_call_trigger"] = 0.60
    if "force_sell_trigger" not in df.columns:
        df["force_sell_trigger"] = 0.70
    return df, f"Loaded V3 Margin_Book from {find_master_workbook(workbook_path)} ({len(df):,} rows)."


def load_liquidity_risk(workbook_path: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, Any], str]:
    df = read_v3_sheet("Liquidity_Risk", workbook_path)
    if df.empty:
        return df, {"liquidity_buffer_ratio": 1.35, "stressed_liquid_assets_bn": 0, "stressed_outflows_bn": 0}, "No Liquidity_Risk sheet loaded."
    df = df.rename(columns={
        "Item": "item",
        "Amount (VND bn)": "amount_bn",
        "Haircut / Stress": "haircut",
        "Stressed Amount (VND bn)": "stressed_amount_bn",
        "Notes": "notes",
    })
    for c in ["amount_bn", "haircut", "stressed_amount_bn"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    item = df.get("item", pd.Series(dtype=str)).astype(str).str.lower()
    assets_mask = item.str.contains("cash|deposit|receivable|liquid|credit line|available", regex=True)
    outflow_mask = item.str.contains("payable|funding|outflow|settlement|debt|call", regex=True)
    stressed_assets = float(df.loc[assets_mask, "stressed_amount_bn"].sum()) if "stressed_amount_bn" in df.columns else 0.0
    stressed_outflows = abs(float(df.loc[outflow_mask, "stressed_amount_bn"].sum())) if "stressed_amount_bn" in df.columns else 0.0
    if stressed_outflows == 0:
        # Conservative placeholder if user has not split asset/outflow categories yet.
        stressed_outflows = max(float(df.get("amount_bn", pd.Series([0])).sum()) * 0.45, 1.0)
    ratio = stressed_assets / stressed_outflows if stressed_outflows else 1.35
    metrics = {
        "liquidity_buffer_ratio": ratio,
        "stressed_liquid_assets_bn": stressed_assets,
        "stressed_outflows_bn": stressed_outflows,
    }
    return df, metrics, f"Loaded V3 Liquidity_Risk ({len(df):,} rows)."


def load_operational_risk(workbook_path: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, Any], str]:
    df = read_v3_sheet("Operational_Risk", workbook_path)
    if df.empty:
        return df, {"failed_settlement_cases": 0, "overdue_exceptions": 0, "monthly_incident_count": 0}, "No Operational_Risk sheet loaded."
    df = df.rename(columns={
        "Date": "date",
        "Risk Event": "risk_event",
        "Category": "category",
        "Business Unit": "business_unit",
        "Impact (VND mn)": "impact_mn",
        "Severity": "severity",
        "Status": "status",
        "Root Cause / Action": "root_cause_action",
    })
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "impact_mn" in df.columns:
        df["impact_mn"] = pd.to_numeric(df["impact_mn"], errors="coerce")
    cat = df.get("category", pd.Series(dtype=str)).astype(str).str.lower()
    status = df.get("status", pd.Series(dtype=str)).astype(str).str.lower()
    metrics = {
        "monthly_incident_count": int(len(df)),
        "failed_settlement_cases": int(cat.str.contains("settlement").sum()),
        "overdue_exceptions": int(status.str.contains("open|overdue|pending").sum()),
        "high_severity_cases": int(df.get("severity", pd.Series(dtype=str)).astype(str).str.lower().str.contains("high|critical").sum()),
        "operational_loss_mn": float(df.get("impact_mn", pd.Series(dtype=float)).sum()),
    }
    return df, metrics, f"Loaded V3 Operational_Risk ({len(df):,} rows)."
