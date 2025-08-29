# Rev 6 Complete update and Auto-Refresh changed to 15 mins
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import altair as alt
from io import BytesIO

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Live Portfolio Dashboard", layout="wide")
st.title("📈 Live Profit & Loss Dashboard")

# ---------------------- AUTO-REFRESH (optional) ----------------------
try:
    from streamlit_autorefresh import st_autorefresh
    _ = st_autorefresh(interval=15 * 60 * 1000, key="portfolio_autorefresh")  # 15 min
except Exception:
    st.sidebar.caption("Tip: `pip install streamlit-autorefresh` for live refresh (5 min).")

# ---------------------- SIDEBAR INPUTS ----------------------
st.sidebar.header("Upload Portfolio Files")
india_file = st.sidebar.file_uploader("Upload Indian Portfolio CSV", type="csv")
us_file = st.sidebar.file_uploader("Upload US Portfolio CSV", type="csv")
currency_choice = st.sidebar.radio("Display Currency", ["INR", "USD"], index=0)

# ---------------------- FX RATE ----------------------
@st.cache_data(ttl=600)
def get_fx_rate():
    try:
        # Try fast_info first
        fx = yf.Ticker("USDINR=X").fast_info.get("lastPrice", None)
        if fx is None or (isinstance(fx, float) and np.isnan(fx)):
            data = yf.download("USDINR=X", period="1d", interval="5m", progress=False)
            fx = float(data["Close"].dropna().iloc[-1])
        return float(fx)
    except Exception:
        return 83.0  # fallback if Yahoo fails
fx_rate = get_fx_rate()

# ---------------------- HELPERS ----------------------
def load_portfolio(file, is_india=True):
    """
    Expect CSV columns: Ticker, Quantity, AvgCost
    (Do NOT include .NS in Indian tickers; the code appends it.)
    """
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Ticker": "ticker", "Quantity": "shares", "AvgCost": "avg_cost"})
    if is_india:
        df["ticker"] = df["ticker"].astype(str).apply(lambda x: x if x.endswith(".NS") else f"{x}.NS")
        df["currency"] = "INR"
    else:
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["currency"] = "USD"
    return df

@st.cache_data(ttl=300)
def fetch_quotes(symbols):
    """
    Efficient quotes fetch:
    - Batch download for last prices
    - Use fast_info for light fundamentals (no slow .info)
    """
    if not symbols:
        return {}
    quotes = {}

    # 1) Batch download last prices
    prices = None
    try:
        prices = yf.download(symbols, period="1d", interval="1m", progress=False, group_by="ticker", threads=True)
    except Exception:
        pass

    for sym in symbols:
        last_price = np.nan
        try:
            if prices is not None:
                # Single-symbol case returns DataFrame with Close column
                if isinstance(prices, pd.DataFrame) and "Close" in prices.columns:
                    last_price = float(prices["Close"].dropna().iloc[-1])
                else:
                    # Multi-symbol case -> prices[sym]["Close"]
                    last_price = float(prices[sym]["Close"].dropna().iloc[-1])
        except Exception:
            pass

        trailingPE = forwardPE = eps = div_yield = beta = np.nan
        try:
            tk = yf.Ticker(sym)
            finfo = tk.fast_info or {}
            last_price = finfo.get("lastPrice", last_price)
            # Some fields may not exist in fast_info; keep NaN if missing
            eps = finfo.get("trailingEps", np.nan)
            trailingPE = finfo.get("trailingPE", np.nan)
            forwardPE = finfo.get("forwardPE", np.nan)
            beta = finfo.get("beta", np.nan)
            div_yield = finfo.get("dividendYield", np.nan)
        except Exception:
            pass

        quotes[sym] = {
            "last_price": last_price,
            "trailingPE": trailingPE,
            "forwardPE": forwardPE,
            "eps": eps,
            "dividendYield": div_yield,
            "beta": beta,
        }
    return quotes

def process_portfolio(df):
    """Builds per-holding metrics table from uploaded CSV."""
    if df is None or df.empty:
        return pd.DataFrame()
    syms = df["ticker"].dropna().unique().tolist()
    qmap = fetch_quotes(syms)

    rows = []
    for _, r in df.iterrows():
        sym = r["ticker"]
        shares = r["shares"]
        avg_cost = r["avg_cost"]
        q = qmap.get(sym, {})
        last_price = q.get("last_price", np.nan)

        cost = shares * avg_cost if pd.notna(shares) and pd.notna(avg_cost) else np.nan
        mv = shares * last_price if pd.notna(shares) and pd.notna(last_price) else np.nan
        pl = mv - cost if pd.notna(mv) and pd.notna(cost) else np.nan
        pl_pct = (pl / cost * 100) if pd.notna(pl) and pd.notna(cost) and cost > 0 else np.nan

        rows.append({
            "Ticker": sym,
            "Shares": shares,
            "Avg Cost": avg_cost,
            "Cost": cost,
            "Last Price": last_price,
            "Market Value": mv,
            "Unrealized P/L": pl,
            "Unrealized P/L %": pl_pct,
            "Trailing PE": q.get("trailingPE", np.nan),
            "Forward PE": q.get("forwardPE", np.nan),
            "EPS": q.get("eps", np.nan),
            "Dividend Yield": q.get("dividendYield", np.nan),
            "Beta": q.get("beta", np.nan),
        })
    return pd.DataFrame(rows)

def convert_currency(df, to="INR"):
    """Converts Cost & Market Value between INR and USD depending on ticker suffix."""
    if df is None or df.empty:
        return df
    out = df.copy()

    if to == "USD":
        out["Cost"] = out.apply(lambda r: r["Cost"]/fx_rate if str(r["Ticker"]).endswith(".NS") else r["Cost"], axis=1)
        out["Market Value"] = out.apply(lambda r: r["Market Value"]/fx_rate if str(r["Ticker"]).endswith(".NS") else r["Market Value"], axis=1)
    else:  # INR
        out["Cost"] = out.apply(lambda r: r["Cost"] if str(r["Ticker"]).endswith(".NS") else r["Cost"]*fx_rate, axis=1)
        out["Market Value"] = out.apply(lambda r: r["Market Value"] if str(r["Ticker"]).endswith(".NS") else r["Market Value"]*fx_rate, axis=1)

    out["Cost"] = pd.to_numeric(out["Cost"], errors="coerce")
    out["Market Value"] = pd.to_numeric(out["Market Value"], errors="coerce")
    out["Unrealized P/L"] = out["Market Value"] - out["Cost"]
    out["Unrealized P/L %"] = ((out["Unrealized P/L"] / out["Cost"]) * 100).round(2)
    return out

def calc_alpha_portfolio(portfolio_df, benchmark="^GSPC"):
    """
    Annualized alpha of portfolio vs benchmark using daily returns.
    - Portfolio weights = current Market Value weights
    - Uses 1y of daily data, aligns dates, annualizes (×252)
    """
    try:
        if portfolio_df is None or portfolio_df.empty:
            return np.nan

        weights = portfolio_df["Market Value"] / portfolio_df["Market Value"].sum()
        tickers = portfolio_df["Ticker"].tolist()

        prices = yf.download(tickers, period="1y", interval="1d", progress=False)["Close"]
        if isinstance(prices, pd.Series):  # single ticker case
            prices = prices.to_frame(tickers[0])
        rets = prices.pct_change().dropna()
        port_ret = (rets * weights.values).sum(axis=1)

        bench_prices = yf.download(benchmark, period="1y", interval="1d", progress=False)["Close"]
        bench_ret = bench_prices.pct_change().dropna()

        df = pd.concat([port_ret, bench_ret], axis=1).dropna()
        df.columns = ["portfolio", "benchmark"]
        if df.empty:
            return np.nan

        cov = np.cov(df["portfolio"], df["benchmark"])[0, 1]
        var = np.var(df["benchmark"])
        beta = cov / var if var > 0 else np.nan
        alpha_daily = df["portfolio"].mean() - beta * df["benchmark"].mean()
        alpha_annualized = alpha_daily * 252
        return alpha_annualized
    except Exception:
        return np.nan

def portfolio_summary_block(df, title, benchmark):
    if df is None or df.empty:
        return
    total_cost = df["Cost"].sum()
    total_mv = df["Market Value"].sum()
    total_pl = total_mv - total_cost
    total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else np.nan
    alpha = calc_alpha_portfolio(df, benchmark=benchmark)

    st.subheader(title)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost", f"{total_cost:,.2f} {currency_choice}")
    c2.metric("Market Value", f"{total_mv:,.2f} {currency_choice}")
    c3.metric("Unrealized P/L", f"{total_pl:,.2f} {currency_choice}", f"{total_pl_pct:.2f}%")
    c4.metric(f"Alpha (annualized vs {benchmark})", f"{alpha:.4f}")

# ---------------------- BUILD PORTFOLIOS ----------------------
india_df = load_portfolio(india_file, is_india=True) if india_file else None
us_df = load_portfolio(us_file, is_india=False) if us_file else None

india_proc = process_portfolio(india_df) if india_df is not None else None
us_proc = process_portfolio(us_df) if us_df is not None else None

# Convert currency for display & downstream calcs
india_proc = convert_currency(india_proc, to=currency_choice)
us_proc = convert_currency(us_proc, to=currency_choice)

if india_proc is not None and us_proc is not None:
    combined_proc = pd.concat([india_proc, us_proc], ignore_index=True)
elif india_proc is not None:
    combined_proc = india_proc.copy()
elif us_proc is not None:
    combined_proc = us_proc.copy()
else:
    combined_proc = None

# ---------------------- SUMMARIES ----------------------
if combined_proc is not None and not combined_proc.empty:
    portfolio_summary_block(combined_proc, "Overall Portfolio Summary", "^GSPC")
if india_proc is not None and not india_proc.empty:
    portfolio_summary_block(india_proc, "India Portfolio Summary", "^NSEI")
if us_proc is not None and not us_proc.empty:
    portfolio_summary_block(us_proc, "US Portfolio Summary", "^IXIC")

# ---------------------- BENCHMARK + PORTFOLIO PERFORMANCE ----------------------
st.subheader("📊 Portfolio vs Benchmarks")
benchmarks = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "NIFTY50": "^NSEI"}

def get_benchmark_perf(ticker):
    try:
        hist = yf.download(ticker, period="1y", interval="1d", progress=False)
        if hist.empty:
            return pd.Series(dtype=float)
        return hist["Close"].pct_change().add(1).cumprod() - 1
    except Exception:
        return pd.Series(dtype=float)

series_list = []
# Benchmarks
for name, symbol in benchmarks.items():
    s = get_benchmark_perf(symbol)
    if not s.empty:
        s.name = name
        series_list.append(s)

def build_cost_basis_series(df_proc, label):
    """
    Portfolio return series using actual cost basis:
    per-holding series = (Close / Avg Cost_native - 1) * (Cost / TotalCost_displayCurrency)
    Sum of weighted contributions → portfolio return vs cost basis (matches summary).
    """
    if df_proc is None or df_proc.empty:
        return None

    total_cost = df_proc["Cost"].sum()
    if not pd.notna(total_cost) or total_cost <= 0:
        return None

    parts = []
    for _, row in df_proc.iterrows():
        t = row["Ticker"]
        avg_cost_native = row["Avg Cost"]
        weight = (row["Cost"] / total_cost) if pd.notna(row["Cost"]) and total_cost > 0 else 0.0
        if not (pd.notna(avg_cost_native) and avg_cost_native > 0 and weight > 0):
            continue
        try:
            hist = yf.download(t, period="1y", interval="1d", progress=False)
            if hist.empty:
                continue
            contrib = (hist["Close"] / float(avg_cost_native) - 1.0) * float(weight)
            contrib.name = t
            parts.append(contrib)
        except Exception:
            continue

    if not parts:
        return None
    df_aligned = pd.concat(parts, axis=1).fillna(0.0)
    series = df_aligned.sum(axis=1)
    series.name = label
    return series

# India & US portfolio series (based on cost basis)
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

# ---------------------- CONSISTENCY CHECK (Summary vs Chart) ----------------------
def check_alignment(proc_df, series, label):
    if proc_df is None or proc_df.empty or series is None or series.empty:
        return
    tot_cost = proc_df["Cost"].sum()
    tot_mv = proc_df["Market Value"].sum()
    if not (pd.notna(tot_cost) and tot_cost > 0 and pd.notna(tot_mv)):
        return
    summary_ret = (tot_mv - tot_cost) / tot_cost
    chart_ret = series.dropna().iloc[-1]
    st.caption(f"🔎 {label}: Summary = {summary_ret:.2%} | Chart = {chart_ret:.2%} | Δ = {(summary_ret - chart_ret):.2%}")

check_alignment(india_proc, india_series, "India Portfolio")
check_alignment(us_proc, us_series, "US Portfolio")

# ---------------------- TABS: INDIA / US / COMBINED ----------------------
if combined_proc is not None and not combined_proc.empty:
    tabs = st.tabs(["Indian Portfolio", "US Portfolio", "Combined Portfolio"])
    port_map = {"Indian Portfolio": india_proc, "US Portfolio": us_proc, "Combined Portfolio": combined_proc}

    for tab, (name, df_show) in zip(tabs, port_map.items()):
        with tab:
            if df_show is not None and not df_show.empty:
                st.dataframe(df_show, use_container_width=True)
                csv = df_show.to_csv(index=False).encode()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{name.lower().replace(' ','_')}.csv",
                    mime="text/csv",
                    key=f"dl_{name.replace(' ','_')}"
                )
            else:
                st.info(f"No data in {name}.")

# ---------------------- EXCEL EXPORT (All) ----------------------
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
        key="excel_all"
    )
