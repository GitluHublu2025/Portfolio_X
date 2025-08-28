# -*- coding: utf-8 -*- TOTAL CODE REPLACED REV 5
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
        if fx is None or (isinstance(fx, float) and np.isnan(fx)):
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
    """Fetch last prices & a few fundamentals for multiple tickers."""
    if not symbols:
        return {}
    quotes = {}
    try:
        data = yf.download(symbols, period="1d", interval="1m", progress=False,
                           group_by='ticker', threads=True)
    except Exception:
        data = None

    def last_price_from(df_like, sym_is_str):
        try:
            if df_like is None:
                return np.nan
            if sym_is_str:  # single symbol case
                if "Close" in df_like:
                    return float(df_like["Close"].dropna().iloc[-1])
                return float(df_like.dropna().iloc[-1])
            # multi
            return float(df_like["Close"].dropna().iloc[-1])
        except Exception:
            return np.nan

    for sym in symbols if isinstance(symbols, list) else [symbols]:
        lp = np.nan
        try:
            if data is not None:
                if isinstance(symbols, str):
                    lp = last_price_from(data, True)
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

        trailingPE = forwardPE = eps = divYield = beta = np.nan
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
    """Calculate alpha of the portfolio vs a benchmark using numpy."""
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
    if df is None or df.empty:
        return df
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
    st.metric(f"Alpha (vs {benchmark})", f"{alpha:.4f}")

# Portfolio summaries with Alpha
if combined_proc is not None and not combined_proc.empty:
    show_summary(combined_proc, "Overall Portfolio Summary", "^GSPC")
if india_proc is not None and not india_proc.empty:
    show_summary(india_proc, "India Portfolio Summary", "^NSEI")
if us_proc is not None and not us_proc.empty:
    show_summary(us_proc, "US Portfolio Summary", "^IXIC")

# ---------------------- BENCHMARK + PORTFOLIO PERFORMANCE ----------------------
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

# Benchmarks
series_list = []
for name, symbol in benchmarks.items():
    s = get_perf(symbol)
    if not s.empty:
        s.name = name
        series_list.append(s)

def build_cost_basis_series(df_proc, label):
    """
    Build a portfolio return series using actual cost basis:
    per-holding series = (Close / Avg Cost_native - 1) * (Cost / TotalCost).
    Weights use 'Cost' (already converted to the display currency);
    Avg Cost uses 'Avg Cost' (native currency), matching Yahoo close currency.
    """
    if df_proc is None or df_proc.empty:
        return None

    total_cost = df_proc["Cost"].sum()
    if total_cost <= 0:
        return None

    parts = []
    for _, row in df_proc.iterrows():
        ticker = row["Ticker"]
        shares = row["Shares"]
        avg_cost_native = row["Avg Cost"]  # native currency (INR for .NS, USD for US)
        weight = (row["Cost"] / total_cost) if pd.notna(row["Cost"]) and total_cost > 0 else 0.0

        if pd.isna(avg_cost_native) or avg_cost_native <= 0 or weight <= 0 or pd.isna(shares) or shares <= 0:
            continue

        try:
            hist = yf.download(ticker, period="1y", interval="1d", progress=False)
            if hist.empty:
                continue
            # holding return vs its cost basis, scaled by weight
            contrib = (hist["Close"] / float(avg_cost_native) - 1.0) * float(weight)
            contrib.name = ticker
            parts.append(contrib)
        except Exception:
            continue

    if not parts:
        return None

    df_aligned = pd.concat(parts, axis=1).fillna(0.0)
    series = df_aligned.sum(axis=1)  # already weighted; represents portfolio return vs cost basis
    series.name = label
    return series

# India & US portfolio series (cost-basis consistent with summary)
india_series = build_cost_basis_series(india_proc, "India Portfolio")
us_series = build_cost_basis_series(us_proc, "US Portfolio")
if india_series is not None:
    series_list.append(india_series)
if us_series is not None:
    series_list.append(us_series)

# Plot
if series_list:
    chart_df = pd.concat(series_list, axis=1).reset_index().rename(columns={"index": "Date"})
    long_df = chart_df.melt(id_vars="Date", var_name="Series", value_name="Return")

    chart = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x="Date:T",
            y=alt.Y("Return:Q", axis=alt.Axis(format="%")),
            color="Series:N",
            tooltip=["Date:T", "Series:N", alt.Tooltip("Return:Q", format=".2%")],
        )
        .properties(title="Benchmark & Portfolio Performance (1Y)")
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No benchmark/portfolio data available right now.")

# ---------------------- PORTFOLIO TABS ----------------------
if combined_proc is not None and not combined_proc.empty:
    tabs = st.tabs(["Indian Portfolio", "US Portfolio", "Combined Portfolio"])
    portfolios = {"Indian Portfolio": india_proc, "US Portfolio": us_proc, "Combined Portfolio": combined_proc}

    for tab, (name, df) in zip(tabs, portfolios.items()):
        with tab:
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
                # CSV download
                csv = df.to_csv(index=False).encode()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{name.lower().replace(' ','_')}.csv",
                    mime="text/csv",
                )

# ---------------------- EXCEL EXPORT ----------------------
if combined_proc is not None and not combined_proc.empty:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        if india_proc is not None and not india_proc.empty:
            india_proc.to_excel(writer, sheet_name="India", index=False)
        if us_proc is not None and not us_proc.empty:
            us_proc.to_excel(writer, sheet_name="US", index=False)
        combined_proc.to_excel(writer, sheet_name="Combined", index=False)
    st.sidebar.download_button(
        "Download Excel (All Portfolios)",
        data=output.getvalue(),
        file_name="portfolio_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
# ---------------------- CONSISTENCY CHECK ----------------------REV 5
def check_alignment(proc_df, series, label):
    if proc_df is None or proc_df.empty or series is None or series.empty:
        return
    # Summary return
    total_cost = proc_df["Cost"].sum()
    total_mv = proc_df["Market Value"].sum()
    summary_ret = (total_mv - total_cost) / total_cost

    # Chart return (last available point)
    chart_ret = series.dropna().iloc[-1]

    st.caption(
        f"🔎 {label}: Summary return = {summary_ret:.2%}, "
        f"Chart return = {chart_ret:.2%}, "
        f"Difference = {(summary_ret - chart_ret):.2%}"
    )

check_alignment(india_proc, india_series, "India Portfolio")
check_alignment(us_proc, us_series, "US Portfolio")




