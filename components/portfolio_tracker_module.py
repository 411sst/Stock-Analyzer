# components/portfolio_tracker_module.py
import yfinance as yf
from utils.indian_stocks import INDIAN_STOCKS
import streamlit as st
import pandas as pd
from utils.data_fetcher import calculate_portfolio_value
from utils.portfolio_manager import add_to_portfolio, remove_from_portfolio, export_to_csv

def load_sample_portfolio():
    """Load sample portfolio data"""
    return [
        {"symbol": "RELIANCE.NS", "quantity": 10, "buy_price": 2500, "buy_date": "2023-06-15"},
        {"symbol": "TCS.NS", "quantity": 5, "buy_price": 3200, "buy_date": "2023-07-20"},
        {"symbol": "INFY.NS", "quantity": 8, "buy_price": 1450, "buy_date": "2023-09-10"}
    ]

def portfolio_tracker_page():
    st.header("Portfolio Tracker")
    
    # Initialize session state for portfolio if not exists
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = load_sample_portfolio()
    
    # Portfolio Actions
    action = st.radio("Select Action", ["View Portfolio", "Add Holding", "Import CSV"])
    
    if action == "Add Holding":
        st.subheader("Add New Holding")
        with st.form("add_holding_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                symbol = st.selectbox("Stock", options=list(INDIAN_STOCKS.keys()))
            
            with col2:
                quantity = st.number_input("Quantity", min_value=1, value=1)
            
            col3, col4 = st.columns(2)
            
            with col3:
                buy_price = st.number_input("Buy Price (₹)", min_value=0.0, value=100.0)
            
            with col4:
                buy_date = st.date_input("Buy Date")
            
            submit_button = st.form_submit_button("Add to Portfolio")
            
            if submit_button:
                new_entry = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "buy_price": buy_price,
                    "buy_date": str(buy_date)
                }
                st.session_state.portfolio = add_to_portfolio(st.session_state.portfolio, new_entry)
                st.success(f"Added {quantity} shares of {INDIAN_STOCKS[symbol]} to your portfolio!")
                st.rerun()
    
    elif action == "Import CSV":
        st.subheader("Import Portfolio from CSV")
        st.info("Upload a CSV file with columns: symbol, quantity, buy_price, buy_date")
        
        # Show sample format
        with st.expander("View Sample CSV Format"):
            sample_df = pd.DataFrame([
                {"symbol": "RELIANCE.NS", "quantity": 10, "buy_price": 2500, "buy_date": "2023-06-15"},
                {"symbol": "TCS.NS", "quantity": 5, "buy_price": 3200, "buy_date": "2023-07-20"}
            ])
            st.dataframe(sample_df)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_columns = ["symbol", "quantity", "buy_price", "buy_date"]
                
                if all(col in df.columns for col in required_columns):
                    st.session_state.portfolio = df.to_dict('records')
                    st.success("Portfolio imported successfully!")
                    st.rerun()
                else:
                    st.error(f"CSV must contain these columns: {', '.join(required_columns)}")
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    # View Portfolio
    st.subheader("Your Portfolio")
    
    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add some holdings to get started!")
        return
    
    # Calculate portfolio value
    portfolio_value = calculate_portfolio_value(st.session_state.portfolio)
    
    # Summary Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Portfolio Value", f"₹{portfolio_value['total_value']:.2f}")
    col2.metric(
        "Profit/Loss", 
        f"₹{portfolio_value['p_and_l']:.2f}", 
        f"{portfolio_value['return_percentage']:.2f}%"
    )
    col3.metric("Benchmark", "NIFTY 50", "+0.66%")
    
    # Portfolio Table
    st.markdown("---")
    st.subheader("Portfolio Holdings")
    
    # Convert portfolio to DataFrame for display
    df = pd.DataFrame(st.session_state.portfolio)
    
    # Get current prices (batched)
    current_prices = {symbol: df[df['symbol'] == symbol]['buy_price'].iloc[0] for symbol in df['symbol'].unique()}
    try:
        symbols = list(df['symbol'].unique())
        data = yf.download(symbols, period="1d", progress=False, group_by='ticker', threads=True)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                close_df = data['Close']
                for sym in close_df.columns:
                    try:
                        current_prices[sym] = float(close_df[sym].dropna().iloc[-1])
                    except Exception:
                        pass
            else:
                for sym in symbols:
                    try:
                        current_prices[sym] = float(data[(sym, 'Close')].dropna().iloc[-1])
                    except Exception:
                        pass
        else:
            # Single symbol case
            try:
                last_close = float(data['Close'].dropna().iloc[-1])
                if len(symbols) == 1:
                    current_prices[symbols[0]] = last_close
            except Exception:
                pass
    except Exception:
        pass
    
    # Add current price and P&L to DataFrame
    df['current_price'] = df['symbol'].map(current_prices)
    df['current_value'] = df['quantity'] * df['current_price']
    df['investment_value'] = df['quantity'] * df['buy_price']
    df['p_and_l'] = df['current_value'] - df['investment_value']
    df['return_percentage'] = (df['p_and_l'] / df['investment_value']) * 100
    
    # Display table with actions
    for i, row in df.iterrows():
        col1, col2, col3, col4, col5, col6, col7 = st.columns([3, 2, 2, 2, 2, 2, 1])
        
        with col1:
            st.write(INDIAN_STOCKS.get(row['symbol'], row['symbol']))
        with col2:
            st.write(f"₹{row['buy_price']:.2f}")
        with col3:
            st.write(f"₹{row['current_price']:.2f}")
        with col4:
            if row['p_and_l'] >= 0:
                st.markdown(f"<span style='color: var(--color-positive); font-family: var(--font-mono); font-weight: 600;'>+₹{row['p_and_l']:.2f}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: var(--color-negative); font-family: var(--font-mono); font-weight: 600;'>₹{row['p_and_l']:.2f}</span>", unsafe_allow_html=True)
        with col5:
            st.write(f"{row['return_percentage']:.1f}%")
        with col6:
            st.write(f"{row['quantity']}")
        with col7:
            if st.button("Delete", key=f"delete_{i}_{row['symbol']}"):
                st.session_state.portfolio = remove_from_portfolio(st.session_state.portfolio, i)
                st.rerun()
    
    # Export Portfolio
    st.markdown("---")
    st.subheader("Export Portfolio")
    csv = export_to_csv(st.session_state.portfolio)
    
    st.download_button(
        label="Download Portfolio as CSV",
        data=csv,
        file_name="my_portfolio.csv",
        mime="text/csv",
    )
    
    # Asset Allocation Chart
    st.markdown("---")
    st.subheader("Asset Allocation")
    
    # Placeholder for pie chart
    st.info("Asset allocation visualization will be added in the next update.")
    
    # Coming Soon Banner
    st.markdown("---")
    st.markdown("""
    <div style='padding: 15px; background-color: var(--color-bg-secondary); border-left: 4px solid var(--color-info); border-radius: 8px; border: 1px solid var(--color-border-subtle); color: var(--color-text-primary);'>
        <strong>Coming Soon:</strong> Auto-sync with brokers functionality for automatic portfolio updates.
    </div>
    """, unsafe_allow_html=True)
