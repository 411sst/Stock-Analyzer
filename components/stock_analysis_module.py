# pages/stock_analysis.py
import streamlit as st
import datetime
import pytz
from utils.data_fetcher import fetch_stock_data, fetch_support_resistance_levels
from utils.technical_analysis import create_candlestick_chart, create_rsi_chart, create_macd_chart, generate_buy_sell_signals
from utils.indian_stocks import INDIAN_STOCKS

def stock_analysis_page():
    st.header("Stock Analysis")
    
    # Stock Selector
    stock_options = {v: k for k, v in INDIAN_STOCKS.items()}
    selected_company = st.selectbox("Select Stock for Analysis", options=list(stock_options.keys()))
    selected_symbol = stock_options[selected_company]
    
    # Date Range Selector
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "Max": "max"
    }
    selected_period = st.selectbox("Select Time Period", options=list(period_options.keys()), index=1)
    
    # Fetch stock data
    with st.spinner("Fetching stock data..."):
        df = fetch_stock_data(selected_symbol, period_options[selected_period])
    
    if df is None or df.empty:
        st.error(f"Could not retrieve data for {selected_symbol}. Please try again later.")
        return
    
    # Current Price Summary
    latest = df.iloc[-1]
    prev_close = df.iloc[-2]['Close'] if len(df) > 1 else df.iloc[0]['Close']
    change = latest['Close'] - prev_close
    percent_change = (change / prev_close) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"₹{latest['Close']:.2f}", f"{change:.2f} ₹")
    col2.metric("Day High", f"₹{latest['High']:.2f}", "+↑" if latest['High'] > prev_close else "-↑")
    col3.metric("Day Low", f"₹{latest['Low']:.2f}", "+↓" if latest['Low'] > prev_close else "-↓")
    col4.metric("Volume", f"{latest['Volume']:,}", f"{(latest['Volume']/latest['Vol_MA']*100):.0f}% of avg")
    
    # Charts
    st.markdown("---")
    st.subheader("Price Chart with Technical Indicators")
    
    fig = create_candlestick_chart(df, f"{selected_company} ({selected_symbol}) Price Chart")
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical Indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Relative Strength Index (RSI)")
        fig_rsi = create_rsi_chart(df)
        st.plotly_chart(fig_rsi, use_container_width=True)
        
    with col2:
        st.subheader("Moving Average Convergence Divergence (MACD)")
        fig_macd = create_macd_chart(df)
        st.plotly_chart(fig_macd, use_container_width=True)
    
    # Support and Resistance Levels
    st.markdown("---")
    st.subheader("Support and Resistance Levels")
    
    supports, resistances = fetch_support_resistance_levels(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Support Levels")
        for i, level in enumerate(supports):
            st.metric(f"Support {i+1}", f"₹{level:.2f}")

    with col2:
        st.markdown("### Resistance Levels")
        for i, level in enumerate(resistances):
            st.metric(f"Resistance {i+1}", f"₹{level:.2f}")
    
    # Buy/Sell Signals
    st.markdown("---")
    st.subheader("Technical Trading Signals")
    
    signals = generate_buy_sell_signals(df, selected_symbol)
    
    if signals:
        for signal in signals:
            confidence = 75  # Default confidence
            
            if "RSI" in signal['reason'] and "MACD" in signal['reason']:
                confidence = 85
            elif "RSI" in signal['reason']:
                confidence = 70
            elif "MACD" in signal['reason']:
                confidence = 70
            
            # Professional signal cards with new color scheme
            signal_class = 'buy' if signal['type'] == 'BUY' else 'sell'
            st.markdown(f"""
            <div class="signal-card {signal_class}">
                <strong style="color: var(--color-text-primary); font-family: var(--font-ui); font-size: 14px; font-weight: 600;">
                    {signal['type']} Signal (Confidence: {confidence}%)
                </strong><br>
                <span style="color: var(--color-text-secondary); font-family: var(--font-ui); font-size: 13px; line-height: 1.6;">
                    {signal['reason']}, RSI(14)={latest['RSI']:.2f} [{ 'Oversold' if latest['RSI'] < 30 else 'Overbought' if latest['RSI'] > 70 else 'Neutral' }],
                    MACD { 'bullish' if latest['MACD'] > latest['MACD_signal'] else 'bearish' } crossover,
                    Volume: {(latest['Volume']/latest['Vol_MA']*100):.0f}% of avg
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No clear buy/sell signals detected based on current technical indicators.")
    
    # Raw Data
    if st.checkbox("Show Raw Data"):
        st.subheader("Historical Data")
        st.dataframe(df.tail(15).style.format({
            'Open': '₹{:.2f}',
            'High': '₹{:.2f}',
            'Low': '₹{:.2f}',
            'Close': '₹{:.2f}',
            'Volume': '{:,.0f}'
        }))