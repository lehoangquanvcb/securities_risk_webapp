# Securities Company CRO Command Center - V3 Logic

This package is designed for a Head of Risk Management role at a securities company.
It uses:

- **vnstock live data** for VNINDEX, HNXINDEX, UPCOMINDEX, VN30 or other symbols.
- **V3 Master Workbook**: `data/securities_risk_management_v3_master.xlsx`.
- **Excel internal placeholders** for margin book, liquidity risk and operational risk.
- **Python risk engine** for VaR, Expected Shortfall, drawdown, margin concentration, liquidity KRI, operational KRI and stress testing.
- **Streamlit dashboard** for CRO / CEO / Risk Committee monitoring.

## Key V3 workbook sheets used by the code

- `Market_Data_vnstock`: market-data structure; app fetches live data via vnstock instead of relying on static rows.
- `Margin_Book`: margin lending exposure, client/ticker/sector concentration.
- `Liquidity_Risk`: liquidity buffer and stressed outflows.
- `Operational_Risk`: incident, settlement, compliance and exception log.
- `Risk_Appetite`, `KRI_Library`, `Risk_Limits`: governance layer.
- `Public_Context`: public macro/market risk inputs and source URLs.
- `Stress_Test`: scenario assumptions.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to GitHub.
2. In Streamlit Cloud, set main file path to:

```text
app.py
```

3. Make sure `data/securities_risk_management_v3_master.xlsx` is included in the repo.

## Replace placeholder data

Replace data in these V3 workbook sheets:

- `Margin_Book`: real margin book from OMS/core system.
- `Liquidity_Risk`: cash, deposits, receivables, payables, settlement obligations, credit lines.
- `Operational_Risk`: trading system incidents, settlement errors, policy exceptions, compliance breaches.
- `Risk_Limits`: approved risk appetite thresholds.
- `Public_Context`: manually update public macro inputs if better official data is available.
