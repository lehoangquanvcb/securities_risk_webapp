# V4 Enterprise market data flow

Copy files:
- app.py -> project root
- src/data_vnstock.py -> replace existing
- update_market_data.py -> project root
- data/raw/market_VNINDEX.csv -> optional starter CSV

Daily update:
```bat
cd /d E:\ABS\securities_risk_webapp
python update_market_data.py
git add data/raw/*.csv
git commit -m "Update market data"
git push origin main
```

Streamlit Cloud reads CSV files, not vnstock directly.
