# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import altair as alt
from datetime import datetime
from io import BytesIO

# ---------------------- AUTO-REFRESH ----------------------
try:
    from streamlit_autorefresh import st_autorefresh
    _ = st_autorefresh(interval=5 * 60 * 1000, key="portfolio_autorefresh")  # 5 min
except Exception:
    st.sidebar.warning(
        'Optional package "streamlit-autorefresh" not installed. '
        'Install with: pip install streamlit-autorefresh'
    )

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Live Portfolio Dashboard", layout="wide")
st.title("📈 Live Profit & Loss Dashboard")

# ---------------------- FILE UPLOAD ----------------------
st.sidebar.header("Upload Portfolio Files")
india_file = st.sidebar.file_uploader("Upload Indian Portfolio CSV", type="csv")
us_file = st.sidebar.file_uploader("Upload US Portfolio CSV", type="csv")

currency_choice = st.sidebar.radio("Display Currency", ["INR", "USD"])

# ---------------------- FX RATE ----------------------
@st.cache_data(ttl=600)
def get_fx_rate():
    try:
        fx = yf.Ticker("USDINR=X").fast_info.get("lastPrice", None)
        if fx is None or np.isnan(fx):
            data = yf.download("USDINR=X", period="1d", interval="5m", progress=False)
            fx = float(data["Close"].dropna().iloc[-1])
        return float(fx)
    except Exception:
        return 83.0  # fallback
fx_rate = get_fx_rate()

# ---------------------- HELPER FUNCTIONS ----------------------
def load_portfolio(file, is_india=True):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Ticker": "ticker", "Quantity": "shares", "AvgCost": "avg_cost"})
    if is_india:
        df["ticker"] = df["ticker"].astype(str).apply(lambda x: x if x.endswith(".NS") else x + ".NS")
        df["currency"] = "INR"
    else:
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["currency"] = "USD"
    return df

@st.cache_data(ttl=300)
def fetch_quotes(symbols):
    """Fetch reliable last prices and fundamentals for multiple tickers."""
    if not symbols:
        return {}
    quotes = {}
    try:
        data = yf.download(symbols, period="1d", interval="1m", progress=False, group_by='ticker', threads=True)
    except Exception:
        data = None

    def last_price(s):
        if s is None or len(s) == 0:
            return np.nan
        if isinstance(s, pd.Series):
            return float(s.dropna().iloc[-1])
        return float(s["Close"].dropna().iloc[-1]) if "Close" in s else np.nan

    for sym in symbols if isinstance(symbols, list) else [symbols]:
        lp = np.nan
        try:
            if data is not None:
                if isinstance(symbols, str):
                    lp = last_price(data["Close"] if "Close" in data else data)
                else:
                    lp = float(data[sym]["Close"].dropna().iloc[-1])
        except Exception:
            pass
        if np.isnan(lp):
            try:
                tk = yf.Ticker(sym)
                lp = tk.fast_info.get("lastPrice", np.nan)
            except Exception:
                lp = np.nan

        trailingPE, forwardPE, eps, divYield, beta = np.nan, np.nan, np.nan, np.nan, np.nan
        try:
            tk = yf.Ticker(sym)
            info = tk.info
            trailingPE = info.get("trailingPE", np.nan)
            forwardPE = info.get("forwardPE", np.nan)
            eps = info.get("trailingEps", np.nan)
            divYield = info.get("dividendYield", np.nan)
            beta = info.get("beta", np.nan)
        except Exception:
            pass

        quotes[sym] = {
            "last_price": lp,
            "trailingPE": trailingPE,
            "forwardPE": forwardPE,
            "eps": eps,
            "dividendYield": divYield,
            "beta": beta,
        }
    return quotes

def calc_alpha_portfolio(portfolio_df, benchmark="^GSPC"):
    """Calculate alpha of whole portfolio vs a benchmark using numpy (no statsmodels)."""
    try:
        if portfolio_df is None or portfolio_df.empty:
            return np.nan

        weights = portfolio_df["Market Value"] / portfolio_df["Market Value"].sum()
        tickers = portfolio_df["Ticker"].tolist()

        prices = yf.download(tickers, period="1y", interval="1d", progress=False)["Close"]
        if isinstance(prices, pd.Series):  # single ticker
            prices = prices.to_frame(tickers[0])
        rets = prices.pct_change().dropna()

        port_ret = (rets * weights.values).sum(axis=1)
        bench = yf.download(benchmark, period="1y", interval="1d", progress=False)["Close"].pct_change().dropna()

        df = pd.concat([port_ret, bench], axis=1).dropna()
        df.columns = ["portfolio", "benchmark"]

        cov = np.cov(df["portfolio"], df["benchmark"])[0, 1]
        var = np.var(df["benchmark"])
        beta = cov / var if var > 0 else np.nan

        alpha = df["portfolio"].mean() - beta * df["benchmark"].mean()
        return alpha
    except Exception:
        return np.nan

def process_portfolio(df):
    if df is None or df.empty:
        return pd.DataFrame()
    tickers = df["ticker"].dropna().unique().tolist()
    quotes = fetch_quotes(tickers)

    results = []
    for _, row in df.iterrows():
        q = quotes.get(row["ticker"], {})
        last_price = q.get("last_price", np.nan)
        cost = row["shares"] * row["avg_cost"]
        mv = row["shares"] * last_price if pd.notna(last_price) else np.nan
        pl = mv - cost if pd.notna(mv) else np.nan

        results.append({
            "Ticker": row["ticker"],
            "Shares": row["shares"],
            "Avg Cost": row["avg_cost"],
            "Cost": cost,
            "Last Price": last_price,
            "Market Value": mv,
            "Unrealized P/L": pl,
            "Unrealized P/L %": (pl / cost * 100) if cost > 0 else np.nan,
            "Trailing PE": q.get("trailingPE", np.nan),
            "Forward PE": q.get("forwardPE", np.nan),
            "EPS": q.get("eps", np.nan),
            "Dividend Yield": q.get("dividendYield", np.nan),
            "Beta": q.get("beta", np.nan),
        })
    return pd.DataFrame(results)

# ---------------------- PROCESS FILES ----------------------
india_df = load_portfolio(india_file, is_india=True) if india_file else None
us_df = load_portfolio(us_file, is_india=False) if us_file else None

india_proc = process_portfolio(india_df) if india_df is not None else None
us_proc = process_portfolio(us_df) if us_df is not None else None

combined_proc = None
if india_proc is not None and us_proc is not None:
    combined_proc = pd.concat([india_proc, us_proc], ignore_index=True)
elif india_proc is not None:
    combined_proc = india_proc.copy()
elif us_proc is not None:
    combined_proc = us_proc.copy()

# ---------------------- CURRENCY CONVERSION ----------------------
def convert_currency(df, to="INR"):
    if df is None:
        return None
    df = df.copy()

    if to == "USD":
        df["Cost"] = df.apply(lambda r: r["Cost"]/fx_rate if str(r["Ticker"]).endswith(".NS") else r["Cost"], axis=1)
        df["Market Value"] = df.apply(lambda r: r["Market Value"]/fx_rate if str(r["Ticker"]).endswith(".NS") else r["Market Value"], axis=1)
    else:  # INR
        df["Cost"] = df.apply(lambda r: r["Cost"] if str(r["Ticker"]).endswith(".NS") else r["Cost"]*fx_rate, axis=1)
        df["Market Value"] = df.apply(lambda r: r["Market Value"] if str(r["Ticker"]).endswith(".NS") else r["Market Value"]*fx_rate, axis=1)

    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
    df["Market Value"] = pd.to_numeric(df["Market Value"], errors="coerce")

    df["Unrealized P/L"] = df["Market Value"] - df["Cost"]
    df["Unrealized P/L %"] = ((df["Unrealized P/L"] / df["Cost"]) * 100).round(2)

    return df

india_proc = convert_currency(india_proc, to=currency_choice)
us_proc = convert_currency(us_proc, to=currency_choice)
combined_proc = convert_currency(combined_proc, to=currency_choice)

# ---------------------- OVERALL SUMMARY ----------------------
def show_summary(df, title, benchmark):
    if df is None or df.empty:
        return
    total_cost = df["Cost"].sum()
    total_mv = df["Market Value"].sum()
    total_pl = total_mv - total_cost
    total_pl_pct = (total_pl/total_cost*100) if total_cost > 0 else np.nan
    alpha = calc_alpha_portfolio(df, benchmark=benchmark)

    st.subheader(title)
    st.metric("Total Cost", f"{total_cost:,.2f} {currency_choice}")
    st.metric("Market Value", f"{total_mv:,.2f} {currency_choice}")
    st.metric("Unrealized P/L", f"{total_pl:,.2f} {currency_choice}", f"{total_pl_pct:.2f}%")
    st.metric("Alpha (vs {})".format(benchmark), f"{alpha:.4f}")

# Portfolio summaries with Alpha
if combined_proc is not None:
    show_summary(combined_proc, "Overall Portfolio Summary", "^GSPC")
if india_proc is not None:
    show_summary(india_proc, "India Portfolio Summary", "^NSEI")
if us_proc is not None:
    show_summary(us_proc, "US Portfolio Summary", "^IXIC")

# ---------------------- BENCHMARK COMPARISON ----------------------
st.subheader("📊 Portfolio vs Benchmarks")
benchmarks = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "NIFTY50": "^NSEI"}

def get_perf(ticker):
    try:
        hist = yf.download(ticker, period="1y", interval="1d", progress=False)
        if hist.empty:
            return pd.Series(dtype=float)
        return hist["Close"].pct_change().add(1).cumprod() - 1
    except Exception:
        return pd.Series(dtype=float)

series_list = []
for name, symbol in benchmarks.items():
    s = get_perf(symbol)
    if not s.empty:
        s.name = name
        series_list.append(s)

if series_list:
    chart_df = pd.concat(series_list, axis=1).reset_index().rename(columns={"index": "Date"})
    long_df = chart_df.melt(id_vars="Date", var_name="Benchmark", value_name="Return")

    chart = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x="Date:T",
            y=alt.Y("Return:Q", axis=alt.Axis(format="%")),
            color="Benchmark:N",
            tooltip=["Date:T", "Benchmark:N", alt.Tooltip("Return:Q", format=".2%")],
        )
        .properties(title="Benchmark Performance (1Y)")
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Benchmark data not available.")

# ---------------------- PORTFOLIO TABS ----------------------
if combined_proc is not None:
    tabs = st.tabs(["Indian Portfolio", "US Portfolio", "Combined Portfolio"])
    portfolios = {"Indian Portfolio": india_proc, "US Portfolio": us_proc, "Combined Portfolio": combined_proc}

    for tab, (name, df) in zip(tabs, portfolios.items()):
        with tab:
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{name.lower().replace(' ','_')}.csv",
                    mime="text/csv",
                )

# ---------------------- EXCEL EXPORT ----------------------
if combined_proc is not None:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        if india_proc is not None:
            india_proc.to_excel(writer, sheet_name="India", index=False)
        if us_proc is not None:
            us_proc.to_excel(writer, sheet_name="US", index=False)
        combined_proc.to_excel(writer, sheet_name="Combined", index=False)
    st.sidebar.download_button(
        "Download Excel (All Portfolios)",
        data=output.getvalue(),
        file_name="portfolio_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
