# 📈 Live P&L Dashboard (Streamlit)

This project provides a **live Profit & Loss dashboard** for tracking Indian and US stock portfolios side by side.  
It is built with **Streamlit**, **Yahoo Finance (`yfinance`)**, and supports **currency conversion (INR/USD)** with auto-refresh.

---

## 🚀 Features
- Upload **two CSV portfolios**:
  - Indian stocks (`.NS` suffix for NSE tickers will be added automatically).
  - US stocks (NYSE/NASDAQ tickers).
- View **Indian portfolio, US portfolio, and Combined portfolio** in separate tabs.
- Live **profit & loss** (in USD or INR, selectable).
- **Overall portfolio summary** with P/L %.
- **Benchmark comparison** vs **S&P 500, NASDAQ, NIFTY50** (1-year cumulative performance).
- **Stock health indicators**:
  - Trailing P/E
  - Forward P/E
  - EPS
  - Dividend Yield
  - Beta
- **EPS projection for 10 years** (assumes 8% CAGR).
- **Auto-refresh every 5 minutes**.
- **Download data**:
  - Per-portfolio CSV
  - All portfolios (India, US, Combined) in one Excel file.

---

## 📂 File Structure
