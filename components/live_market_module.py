import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf

# Cache data to limit API calls; refreshes every 60 seconds
@st.cache_data(ttl=60, show_spinner=False)
def fetch_index(series: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
    try:
        df = yf.download(series, period=period, interval=interval, progress=False)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_many(tickers: list[str], period: str = "1d", interval: str = "1m") -> dict[str, pd.DataFrame]:
    try:
        data = yf.download(
            tickers=" ".join(tickers),
            period=period,
            interval=interval,
            group_by='ticker',
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        result: dict[str, pd.DataFrame] = {}
        # When a single ticker is requested, yfinance returns a plain DataFrame
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                if (t,) in [(lvl[0],) for lvl in data.columns.levels[0].tolist()]:
                    # Safer extraction across yfinance versions
                    try:
                        result[t] = data[t].dropna()
                    except Exception:
                        try:
                            result[t] = data.xs(t, axis=1, level=0, drop_level=False)
                        except Exception:
                            result[t] = pd.DataFrame()
                else:
                    try:
                        result[t] = data[t].dropna()
                    except Exception:
                        result[t] = pd.DataFrame()
        else:
            # Fallback: single ticker shape
            if len(tickers) == 1:
                result[tickers[0]] = data.dropna() if isinstance(data, pd.DataFrame) else pd.DataFrame()
            else:
                for t in tickers:
                    result[t] = pd.DataFrame()
        return result
    except Exception:
        return {t: pd.DataFrame() for t in tickers}


def _pct_change_today(df: pd.DataFrame) -> float:
    try:
        if df is None or df.empty:
            return 0.0
        close = df['Close'].dropna()
        if len(close) < 2:
            return 0.0
        return float((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100.0)
    except Exception:
        return 0.0


def live_market_page():
    st.title("Live Market Overview")
    st.caption("Refresh data is cached for up to 60 seconds to reduce API calls.")

    # Index overview: NIFTY and Bank Nifty (if available)
    idx_cols = st.columns(2)
    indices = {
        "^NSEI": "NIFTY 50",
        "^NSEBANK": "Bank Nifty",
    }

    index_data = {sym: fetch_index(sym, period="1d", interval="1m") for sym in indices.keys()}

    for i, (sym, name) in enumerate(indices.items()):
        with idx_cols[i]:
            df = index_data.get(sym, pd.DataFrame())
            if df is not None and not df.empty:
                current = float(df['Close'].iloc[-1])
                pct = _pct_change_today(df)
                st.metric(name, f"{current:,.0f}", f"{pct:+.2f}%")
            else:
                st.metric(name, "—", "—")

    st.markdown("---")

    # Intraday chart for NIFTY last ~90 minutes (business minutes based on available data)
    nifty_df = index_data.get("^NSEI", pd.DataFrame())
    if not nifty_df.empty:
        # Take last ~90 rows if 1m interval
        plot_df = nifty_df.tail(90).copy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df['Close'].astype(float),
                mode='lines',
                name='NIFTY 50',
                line=dict(color='#3b82f6', width=3),
                hovertemplate='Time: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
            )
        )
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            template='plotly_dark',
            xaxis=dict(gridcolor='#242424', showline=False, zeroline=False),
            yaxis=dict(gridcolor='#242424', showline=False, zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Top movers (from a curated list to keep calls light)
    curated = [
        'RELIANCE.NS','TCS.NS','INFY.NS','HDFCBANK.NS','ICICIBANK.NS','SBIN.NS','KOTAKBANK.NS','ITC.NS',
        'HINDUNILVR.NS','LT.NS','AXISBANK.NS','HCLTECH.NS','WIPRO.NS','ASIANPAINT.NS','BAJFINANCE.NS',
        'MARUTI.NS','ULTRACEMCO.NS','SUNPHARMA.NS','BHARTIARTL.NS','TITAN.NS'
    ]
    movers = fetch_many(curated, period="1d", interval="1m")

    rows = []
    for sym, df in movers.items():
        if df is not None and not df.empty and 'Close' in df:
            chg = _pct_change_today(df)
            last = float(df['Close'].iloc[-1]) if not df['Close'].empty else np.nan
            rows.append({"symbol": sym, "last": last, "change": chg})

    if rows:
        mv = pd.DataFrame(rows).dropna().sort_values('change', ascending=False)
        top_gainers = mv.head(5)
        top_losers = mv.tail(5).sort_values('change')

        g_col, l_col = st.columns(2)
        with g_col:
            st.subheader("Top Gainers")
            st.dataframe(
                top_gainers.assign(
                    last=lambda d: d['last'].map(lambda v: f"₹{v:,.2f}"),
                    change=lambda d: d['change'].map(lambda v: f"{v:+.2f}%")
                ).rename(columns={"symbol":"Symbol","last":"Last","change":"Change"}),
                use_container_width=True,
            )
        with l_col:
            st.subheader("Top Losers")
            st.dataframe(
                top_losers.assign(
                    last=lambda d: d['last'].map(lambda v: f"₹{v:,.2f}"),
                    change=lambda d: d['change'].map(lambda v: f"{v:+.2f}%")
                ).rename(columns={"symbol":"Symbol","last":"Last","change":"Change"}),
                use_container_width=True,
            )
    else:
        st.info("Live movers unavailable right now. Please try refreshing.")

    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("Refresh now", type="secondary"):
        # Clearing caches will force a refetch on rerun
        fetch_index.clear()
        fetch_many.clear()
        st.experimental_rerun()
