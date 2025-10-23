# pages/market_overview.py
from datetime import datetime
import pytz
import streamlit as st
import plotly.express as px
from utils.data_fetcher import get_market_status, fetch_index_data, fetch_gainers_losers, fetch_sector_performance
from utils.indian_stocks import INDIAN_STOCKS

def market_overview_page():
    st.header("Indian Market Overview")
    
    # Market Status
    is_market_open, current_time = get_market_status()
    status_color_var = "var(--color-positive)" if is_market_open else "var(--color-negative)"
    status_text = "Open" if is_market_open else "Closed"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="
            padding: 10px; 
            border-radius: 8px; 
            background-color: var(--color-bg-secondary); 
            color: var(--color-text-primary);
            border-left: 4px solid {status_color_var};
        ">
            <strong>Market Status:</strong> {status_text}<br>
            <small>IST Time: {current_time}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Indices
    index_data = fetch_index_data()
    
    if index_data:
        cols = st.columns(len(index_data))
        for i, (ticker, data) in enumerate(index_data.items()):
            with cols[i]:
                st.metric(
                    label=f"{data['name']} ({ticker})",
                    value=f"₹{data['value']:.2f}",
                    delta=f"{data['percent_change']:.2f}% {'↑' if data['percent_change'] > 0 else '↓'}"
                )
    else:
        st.warning("Unable to fetch index data due to rate limiting. Please try again later.")
        # Show mock data
        col1, col2 = st.columns(2)
        with col1:
            st.metric("NIFTY 50", "₹25,400.00", "0.5%")
        with col2:
            st.metric("SENSEX", "₹83,500.00", "0.7%")    
    st.markdown("---")
    
    # Top Gainers and Losers
    gainers, losers = fetch_gainers_losers()
    
    st.subheader("Top Gainers and Losers")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Gainers")
        for gainer in gainers:
            st.markdown(f"- {INDIAN_STOCKS.get(gainer['symbol'], gainer['symbol'])} (₹{gainer['price']:.2f}) (+{gainer['change']:.2f}% ↑)")
    
    with col2:
        st.markdown("### Top Losers")
        for loser in losers:
            st.markdown(f"- {INDIAN_STOCKS.get(loser['symbol'], loser['symbol'])} (₹{loser['price']:.2f}) (-{loser['change']:.2f}% ↓)")
    
    st.markdown("---")
    
    # Sector Performance
    sector_data = fetch_sector_performance()
    
    st.subheader("Sector Performance")
    sector_names = [item['sector'] for item in sector_data]
    sector_perf = [item['performance'] for item in sector_data]
    
    colors = ['var(--color-positive)' if perf > 0 else 'var(--color-negative)' for perf in sector_perf]
    
    fig = px.bar(
        x=sector_names,
        y=sector_perf,
        color=colors,
        color_discrete_map="identity",
        labels={'x': 'Sector', 'y': 'Performance (%)'},
        title="Sector Performance"
    )
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#0F0F0F',
        paper_bgcolor='#0F0F0F',
        font=dict(color='var(--color-text-primary)'),
        xaxis=dict(gridcolor='#242424'),
        yaxis=dict(gridcolor='#242424'),
        hoverlabel=dict(bgcolor='#1A1A1A', bordercolor='#404040')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # USD/INR Impact
    st.subheader("USD/INR Exchange Rate Impact")
    impact_cols = st.columns(4)
    
    sectors = ["Banking", "IT", "Pharma", "Energy"]
    impacts = ["Positive", "Positive", "Negative", "Negative"]
    
    for i, (sector, impact) in enumerate(zip(sectors, impacts)):
        with impact_cols[i]:
            st.markdown(f"""
            <div style="
                border: 1px solid var(--color-border-subtle); 
                padding: 10px; 
                border-radius: 8px; 
                margin-bottom: 10px;
                background-color: var(--color-bg-secondary);
                color: var(--color-text-primary);
            ">
                <strong>{sector}</strong><br>
                <small>Impact: <span style='color: {"var(--color-positive)" if impact == 'Positive' else "var(--color-negative)"};'>{impact}</span></small><br>
                <small>Exposure: 25%</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Advanced Market Data (always shown; global modes removed)
    st.markdown("---")
    st.subheader("Advanced Market Data")
    st.write("Professional traders can access additional data like:")
    st.write("- Order book depth")
    st.write("- Institutional buying/selling patterns")
    st.write("- Options chain analysis")
    st.write("- Derivatives data")
