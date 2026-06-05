from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_vnstock import load_market_data, load_multiple_market_data
from src.excel_io import (
    MASTER_WORKBOOK,
    find_master_workbook,
    load_kri_library,
    load_liquidity_risk,
    load_margin_book,
    load_operational_risk,
    load_risk_appetite,
    load_risk_limits,
    load_stress_template,
)
from src.public_context import latest_public_context, load_public_context
from src.risk_engine import (
    build_kri_table,
    compute_margin_metrics,
    compute_market_metrics,
    overall_status,
    stress_test,
)

st.set_page_config(page_title="V3 Securities CRO Command Center", layout="wide")
st.title("Securities Company CRO Command Center - V3")
st.caption("V3 logic: vnstock live market data + V3 Master Workbook inputs + public context + KRI / VaR / Stress Test")

with st.sidebar:
    st.header("0) V3 Master Workbook")
    workbook_path_text = st.text_input("Workbook path", str(MASTER_WORKBOOK))
    try:
        workbook_path = find_master_workbook(workbook_path_text)
        st.success(f"Workbook loaded: {workbook_path}")
    except Exception as exc:
        workbook_path = Path(workbook_path_text)
        st.error(str(exc))

    st.header("1) vnstock market data")
    symbol = st.text_input("Main market symbol", "VNINDEX")
    watchlist = st.text_input("Watchlist symbols", "")
    source = st.selectbox("vnstock source", ["VCI"], index=0)
    start_date = st.date_input("Start date", value=date(2024, 1, 1))
    end_date = st.date_input("End date", value=date.today())
    strict_live = st.checkbox("Require live vnstock data only", value=False)
    refresh = st.button("Refresh market data")
    st.caption("Tip: avoid repeated refreshes; vnstock free API may rate-limit requests.")

    st.header("2) Portfolio assumption")
    prop_trading_value = st.number_input("Proprietary trading portfolio value (VND)", value=120_000_000_000, step=10_000_000_000)
    total_margin_limit = st.number_input("Total margin limit (VND)", value=500_000_000_000, step=10_000_000_000)

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cached_market(symbol: str, start: str, end: str, source: str, allow_sample_fallback: bool):
    return load_market_data(symbol=symbol, start=start, end=end, source=source, allow_sample_fallback=allow_sample_fallback)

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cached_watchlist(symbols: tuple[str, ...], start: str, end: str, source: str):
    return load_multiple_market_data(symbols=symbols, start=start, end=end, source=source)

if refresh:
    cached_market.clear()
    cached_watchlist.clear()

start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")

market_df, market_msg = cached_market(symbol.strip().upper(), start, end, source, not strict_live)
margin_df, margin_msg = load_margin_book(workbook_path)
liquidity_df, liquidity_metrics, liquidity_msg = load_liquidity_risk(workbook_path)
operational_df, operational_metrics, operational_msg = load_operational_risk(workbook_path)
public_context = latest_public_context(load_public_context(workbook_path=workbook_path))
risk_limits = load_risk_limits(workbook_path)
risk_appetite = load_risk_appetite(workbook_path)
kri_library = load_kri_library(workbook_path)
stress_template = load_stress_template(workbook_path)

for msg in [market_msg, margin_msg, liquidity_msg, operational_msg]:
    if "LIVE vnstock" in msg:
        st.success(msg)
    elif "SAMPLE" in msg or "No " in msg:
        st.warning(msg)
    else:
        st.caption(msg)

market = compute_market_metrics(market_df, portfolio_value_vnd=prop_trading_value)
margin = compute_margin_metrics(margin_df, total_margin_limit)
kri = build_kri_table(market, margin, liquidity_metrics, operational_metrics, public_context)
stress = stress_test(margin["total_exposure"], prop_trading_value=prop_trading_value, scenarios=stress_template)
status = overall_status(kri)
red_count = int((kri["status"] == "Red").sum())
amber_count = int((kri["status"] == "Amber").sum())

status_fn = {"Green": st.success, "Amber": st.warning, "Red": st.error}.get(status, st.info)
status_fn(f"Overall Risk Status: {status}")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "1. CRO Dashboard",
    "2. Market Risk / VaR",
    "3. Margin Risk",
    "4. Liquidity Risk",
    "5. Operational & Compliance",
    "6. Risk Appetite & KRI",
    "7. Stress Testing",
    "8. Risk Committee Pack",
])

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{symbol.upper()} latest", f"{market['latest_close']:,.0f}", f"{market['return_1d']:.2%}")
    c2.metric("20D volatility", f"{market['vol_20d']:.2%}")
    c3.metric("Hist. VaR 95%", f"{market['var_95_1d']:.2%}")
    c4.metric("Margin utilization", f"{margin['margin_utilization']:.1%}")
    c5.metric("Red / Amber alerts", f"{red_count} / {amber_count}")

    st.subheader("Board-level KRI Summary")
    st.dataframe(kri, use_container_width=True)

    st.subheader("Immediate Risk Actions")
    risky = kri[kri["status"].isin(["Red", "Amber"])]
    if risky.empty:
        st.success("No Red/Amber KRI. Maintain normal monitoring.")
    for _, row in risky.iterrows():
        text = f"{row['risk_area']}: {row['indicator']} = {row['current']:.4g}. Source: {row['source']}"
        if row["status"] == "Red":
            st.error(text + " — escalate to CEO/Risk Committee and freeze incremental exposure if needed.")
        else:
            st.warning(text + " — increase monitoring and prepare mitigation plan.")

with tab2:
    st.subheader("vnstock Market Trend")
    fig = px.line(market["data"], x="date", y=["close", "ma_20", "ma_50"], title=f"{symbol.upper()} close and moving averages")
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1D return", f"{market['return_1d']:.2%}")
    c2.metric("5D return", f"{market['return_5d']:.2%}")
    c3.metric("Drawdown from peak", f"{market['drawdown_from_peak']:.2%}")
    c4.metric("ES 95%", f"{market['expected_shortfall_95_1d']:.2%}")
    var_table = pd.DataFrame({
        "method": ["Historical VaR 95%", "Historical VaR 99%", "Parametric VaR 95%", "Parametric VaR 99%", "Expected Shortfall 95%", "Expected Shortfall 99%"],
        "daily_return_loss": [market["var_95_1d"], market["var_99_1d"], market["parametric_var_95_1d"], market["parametric_var_99_1d"], market["expected_shortfall_95_1d"], market["expected_shortfall_99_1d"]],
    })
    var_table["estimated_loss_vnd"] = var_table["daily_return_loss"].abs() * prop_trading_value
    st.dataframe(var_table, use_container_width=True)
    st.subheader("Watchlist")
    symbols = tuple([s.strip().upper() for s in watchlist.split(",") if s.strip()])

    if not symbols:
        st.info("Watchlist is disabled by default to avoid vnstock API rate limits on Streamlit Cloud. Enter symbols manually, e.g. VNINDEX, when needed.")
    else:
        watch_df, watch_msgs = cached_watchlist(symbols, start, end, source)
        for msg in watch_msgs:
            st.caption(msg)
        if not watch_df.empty:
            latest = watch_df.sort_values("date").groupby("symbol", as_index=False).tail(1)
            st.dataframe(latest[["symbol", "date", "close", "data_source"]], use_container_width=True)
            st.plotly_chart(px.line(watch_df, x="date", y="close", color="symbol", title="Watchlist close prices"), use_container_width=True)

with tab3:
    st.subheader("Margin Book from V3 Master Workbook")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total exposure", f"{margin['total_exposure']/1e9:,.0f} VND bn")
    c2.metric("Top 1 concentration", f"{margin['top1_concentration']:.1%}")
    c3.metric("Top 10 concentration", f"{margin['top10_concentration']:.1%}")
    c4.metric("Force-sell accounts", f"{margin['num_force_sell']}")
    if not margin["data"].empty and "ticker" in margin["data"].columns:
        by_ticker = margin["data"].groupby("ticker", as_index=False)["exposure_vnd"].sum().sort_values("exposure_vnd", ascending=False)
        st.plotly_chart(px.bar(by_ticker, x="ticker", y="exposure_vnd", title="Margin exposure by ticker"), use_container_width=True)
    if not margin["data"].empty and "sector" in margin["data"].columns:
        by_sector = margin["data"].groupby("sector", as_index=False)["exposure_vnd"].sum().sort_values("exposure_vnd", ascending=False)
        st.plotly_chart(px.bar(by_sector, x="sector", y="exposure_vnd", title="Margin exposure by sector"), use_container_width=True)
    st.dataframe(margin["data"].sort_values("exposure_vnd", ascending=False).head(50), use_container_width=True)

with tab4:
    st.subheader("Liquidity Risk from V3 Master Workbook")
    c1, c2, c3 = st.columns(3)
    c1.metric("Liquidity buffer ratio", f"{liquidity_metrics['liquidity_buffer_ratio']:.2f}x")
    c2.metric("Stressed liquid assets", f"{liquidity_metrics['stressed_liquid_assets_bn']:,.0f} VND bn")
    c3.metric("Stressed outflows", f"{liquidity_metrics['stressed_outflows_bn']:,.0f} VND bn")
    st.dataframe(liquidity_df, use_container_width=True)

with tab5:
    st.subheader("Operational Risk & Compliance from V3 Master Workbook")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidents", f"{operational_metrics['monthly_incident_count']}")
    c2.metric("High severity", f"{operational_metrics['high_severity_cases']}")
    c3.metric("Open exceptions", f"{operational_metrics['overdue_exceptions']}")
    c4.metric("Loss amount", f"{operational_metrics['operational_loss_mn']:,.0f} VND mn")
    if not operational_df.empty and "category" in operational_df.columns:
        st.plotly_chart(px.bar(operational_df.groupby("category", as_index=False).size(), x="category", y="size", title="Incidents by category"), use_container_width=True)
    st.dataframe(operational_df, use_container_width=True)

with tab6:
    st.subheader("Risk Appetite Statement")
    st.dataframe(risk_appetite, use_container_width=True)
    st.subheader("KRI Library")
    st.dataframe(kri_library, use_container_width=True)
    st.subheader("Risk Limits")
    st.dataframe(risk_limits, use_container_width=True)
    st.subheader("Public / Macro Context")
    st.dataframe(public_context, use_container_width=True)

with tab7:
    st.subheader("Stress Testing")
    st.dataframe(stress, use_container_width=True)
    st.plotly_chart(px.bar(stress, x="scenario", y="total_estimated_loss_vnd", title="Estimated loss by scenario"), use_container_width=True)

with tab8:
    st.subheader("Risk Committee Pack - Download Center")
    st.caption("These CSVs can be attached to the Risk Committee report or pasted back into the V3 Master Workbook.")
    st.download_button("Download KRI CSV", kri.to_csv(index=False).encode("utf-8"), file_name="risk_kri_v3.csv")
    st.download_button("Download Stress Test CSV", stress.to_csv(index=False).encode("utf-8"), file_name="stress_test_v3.csv")
    st.download_button("Download Market Data CSV", market["data"].to_csv(index=False).encode("utf-8"), file_name="market_data_vnstock.csv")
    st.download_button("Download Margin Book CSV", margin["data"].to_csv(index=False).encode("utf-8"), file_name="margin_book_v3.csv")
    st.download_button("Download Liquidity CSV", liquidity_df.to_csv(index=False).encode("utf-8"), file_name="liquidity_v3.csv")
    st.download_button("Download Operational Risk CSV", operational_df.to_csv(index=False).encode("utf-8"), file_name="operational_risk_v3.csv")
