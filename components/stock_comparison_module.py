# components/stock_comparison_module.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from utils.indian_stocks import INDIAN_STOCKS
from utils.chart_config import get_semantic_colors


def fetch_comparison_data(symbols: list, period: str = "1y") -> dict:
    """Fetch historical data for multiple stocks"""
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if not df.empty:
                data[symbol] = df
        except Exception:
            data[symbol] = pd.DataFrame()
    return data


def calculate_normalized_returns(data: dict) -> pd.DataFrame:
    """Calculate normalized returns for comparison (base 100)"""
    normalized = pd.DataFrame()

    for symbol, df in data.items():
        if not df.empty and 'Close' in df.columns:
            # Normalize to base 100
            base_price = df['Close'].iloc[0]
            normalized[symbol] = (df['Close'] / base_price) * 100

    return normalized


def calculate_comparison_metrics(data: dict) -> pd.DataFrame:
    """Calculate comparison metrics for all stocks"""
    metrics = []

    for symbol, df in data.items():
        if df.empty or 'Close' not in df.columns:
            continue

        close = df['Close']

        # Calculate metrics
        current_price = close.iloc[-1]
        start_price = close.iloc[0]
        total_return = ((current_price - start_price) / start_price) * 100

        # Volatility (annualized)
        daily_returns = close.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100

        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

        # Max drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100
        max_drawdown = drawdown.min()

        # Average volume
        avg_volume = df['Volume'].mean() if 'Volume' in df.columns else 0

        metrics.append({
            'Symbol': symbol,
            'Name': INDIAN_STOCKS.get(symbol, symbol),
            'Current Price': f"₹{current_price:.2f}",
            'Total Return': f"{total_return:.2f}%",
            'Volatility': f"{volatility:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown:.2f}%",
            'Avg Volume': f"{avg_volume:,.0f}",
            '_total_return': total_return,  # For sorting
            '_volatility': volatility,
            '_sharpe': sharpe
        })

    return pd.DataFrame(metrics)


def create_comparison_chart(normalized_data: pd.DataFrame, symbols: list) -> go.Figure:
    """Create synchronized comparison chart with normalized returns"""
    colors = get_semantic_colors()

    # Define color palette for multiple stocks
    stock_colors = [
        colors['accent'],
        '#FFB86C',  # Orange
        '#BD93F9',  # Purple
        '#50FA7B'   # Green
    ]

    fig = go.Figure()

    for i, symbol in enumerate(symbols):
        if symbol in normalized_data.columns:
            fig.add_trace(go.Scatter(
                x=normalized_data.index,
                y=normalized_data[symbol],
                mode='lines',
                name=INDIAN_STOCKS.get(symbol, symbol),
                line=dict(color=stock_colors[i % len(stock_colors)], width=2),
                hovertemplate=f'<b>{INDIAN_STOCKS.get(symbol, symbol)}</b><br>' +
                             'Date: %{x|%Y-%m-%d}<br>' +
                             'Value: %{y:.2f}<extra></extra>'
            ))

    fig.update_layout(
        title='Normalized Price Comparison (Base 100)',
        plot_bgcolor='#1E1B18',
        paper_bgcolor='#1E1B18',
        font=dict(color='#C8C4C9', family='Inter, sans-serif', size=13),
        xaxis=dict(
            title='Date',
            gridcolor='#626C66',
            gridwidth=1,
            showline=False,
            zeroline=False,
            tickfont=dict(color='#9A969B', size=11)
        ),
        yaxis=dict(
            title='Normalized Value',
            gridcolor='#626C66',
            gridwidth=1,
            showline=False,
            zeroline=True,
            zerolinecolor='#626C66',
            zerolinewidth=1,
            tickfont=dict(color='#9A969B', size=11)
        ),
        hoverlabel=dict(
            bgcolor='#2A2622',
            bordercolor='#7A8479',
            font=dict(color='#FFFAFF', family='JetBrains Mono, monospace', size=12)
        ),
        title_font=dict(color='#FFFAFF', size=16, family='Inter, sans-serif'),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(30, 27, 24, 0.8)',
            bordercolor='#626C66',
            borderwidth=1,
            font=dict(color='#C8C4C9', size=11)
        ),
        height=500
    )

    return fig


def create_volume_comparison_chart(data: dict, symbols: list) -> go.Figure:
    """Create volume comparison chart"""
    colors = get_semantic_colors()
    stock_colors = [colors['accent'], '#FFB86C', '#BD93F9', '#50FA7B']

    fig = make_subplots(
        rows=len(symbols), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[INDIAN_STOCKS.get(s, s) for s in symbols]
    )

    for i, symbol in enumerate(symbols, start=1):
        if symbol in data and not data[symbol].empty and 'Volume' in data[symbol].columns:
            df = data[symbol]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name=INDIAN_STOCKS.get(symbol, symbol),
                    marker_color=stock_colors[(i-1) % len(stock_colors)],
                    showlegend=False,
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Volume: %{y:,.0f}<extra></extra>'
                ),
                row=i, col=1
            )

    fig.update_layout(
        plot_bgcolor='#1E1B18',
        paper_bgcolor='#1E1B18',
        font=dict(color='#C8C4C9', family='Inter, sans-serif', size=12),
        height=150 * len(symbols),
        title=dict(
            text='Volume Comparison',
            font=dict(color='#FFFAFF', size=16, family='Inter, sans-serif')
        ),
        hoverlabel=dict(
            bgcolor='#2A2622',
            bordercolor='#7A8479',
            font=dict(color='#FFFAFF', family='JetBrains Mono, monospace', size=12)
        )
    )

    fig.update_xaxes(
        gridcolor='#626C66',
        gridwidth=1,
        showline=False,
        tickfont=dict(color='#9A969B', size=10)
    )

    fig.update_yaxes(
        gridcolor='#626C66',
        gridwidth=1,
        showline=False,
        tickfont=dict(color='#9A969B', size=10)
    )

    return fig


def create_correlation_heatmap(data: dict) -> go.Figure:
    """Create correlation heatmap for selected stocks"""
    returns_df = pd.DataFrame()

    for symbol, df in data.items():
        if not df.empty and 'Close' in df.columns:
            returns_df[INDIAN_STOCKS.get(symbol, symbol)] = df['Close'].pct_change()

    if returns_df.empty:
        return None

    corr_matrix = returns_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[
            [0, '#3B020A'],      # Negative correlation (red)
            [0.5, '#626C66'],    # No correlation (gray)
            [1, '#7FC7B7']       # Positive correlation (teal)
        ],
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont=dict(color='#FFFAFF', family='JetBrains Mono, monospace', size=11),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
        colorbar=dict(
            title=dict(
                text='Correlation',
                side='right',
                font=dict(color='#C8C4C9', size=12)
            ),
            tickmode='linear',
            tick0=-1,
            dtick=0.5,
            tickfont=dict(color='#9A969B', size=10)
        )
    ))

    fig.update_layout(
        title='Return Correlation Matrix',
        plot_bgcolor='#1E1B18',
        paper_bgcolor='#1E1B18',
        font=dict(color='#C8C4C9', family='Inter, sans-serif', size=13),
        xaxis=dict(
            side='bottom',
            tickfont=dict(color='#9A969B', size=11)
        ),
        yaxis=dict(
            tickfont=dict(color='#9A969B', size=11)
        ),
        title_font=dict(color='#FFFAFF', size=16, family='Inter, sans-serif'),
        height=400
    )

    return fig


def stock_comparison_page():
    """Main stock comparison page"""
    st.header("Multi-Stock Comparison")
    st.caption("Compare performance, volatility, and correlation across multiple stocks")

    # Stock selection
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_stocks = st.multiselect(
            "Select stocks to compare (2-4 stocks)",
            options=list(INDIAN_STOCKS.keys()),
            format_func=lambda x: f"{INDIAN_STOCKS[x]} ({x})",
            max_selections=4,
            default=['RELIANCE.NS', 'TCS.NS'] if len(INDIAN_STOCKS) >= 2 else []
        )

    with col2:
        time_period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )

    if len(selected_stocks) < 2:
        st.info("Please select at least 2 stocks to compare.")
        return

    # Fetch data
    with st.spinner("Fetching comparison data..."):
        stock_data = fetch_comparison_data(selected_stocks, period=time_period)

    # Check if we have valid data
    valid_stocks = [s for s in selected_stocks if not stock_data.get(s, pd.DataFrame()).empty]

    if len(valid_stocks) < 2:
        st.error("Unable to fetch data for selected stocks. Please try different stocks or time period.")
        return

    # Calculate normalized returns
    normalized_data = calculate_normalized_returns(stock_data)

    # Performance comparison chart
    st.subheader("Relative Performance")
    comparison_chart = create_comparison_chart(normalized_data, valid_stocks)
    st.plotly_chart(comparison_chart, use_container_width=True)

    # Metrics comparison
    st.markdown("---")
    st.subheader("Key Metrics Comparison")

    metrics_df = calculate_comparison_metrics(stock_data)

    # Display metrics table
    display_df = metrics_df[[
        'Name', 'Current Price', 'Total Return',
        'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Avg Volume'
    ]].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Highlight best/worst performers
    col1, col2, col3 = st.columns(3)

    if not metrics_df.empty:
        best_return = metrics_df.loc[metrics_df['_total_return'].idxmax()]
        worst_return = metrics_df.loc[metrics_df['_total_return'].idxmin()]
        best_sharpe = metrics_df.loc[metrics_df['_sharpe'].idxmax()]

        with col1:
            st.metric(
                "Best Performer",
                best_return['Name'],
                best_return['Total Return']
            )

        with col2:
            st.metric(
                "Worst Performer",
                worst_return['Name'],
                worst_return['Total Return']
            )

        with col3:
            st.metric(
                "Best Risk-Adjusted",
                best_sharpe['Name'],
                f"Sharpe: {best_sharpe['Sharpe Ratio']}"
            )

    # Correlation analysis
    st.markdown("---")
    st.subheader("Correlation Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        corr_heatmap = create_correlation_heatmap(stock_data)
        if corr_heatmap:
            st.plotly_chart(corr_heatmap, use_container_width=True)
        else:
            st.info("Unable to calculate correlations with current data.")

    with col2:
        st.markdown("#### Understanding Correlation")
        st.markdown("""
        **Correlation values range from -1 to +1:**

        - **+1.0**: Perfect positive correlation
          - Stocks move together

        - **0.0**: No correlation
          - Independent movements

        - **-1.0**: Perfect negative correlation
          - Stocks move opposite

        **Portfolio Diversification:**

        Lower correlation between stocks provides better risk diversification.
        """)

    # Volume comparison
    st.markdown("---")
    st.subheader("Volume Analysis")

    with st.expander("View Volume Comparison", expanded=False):
        volume_chart = create_volume_comparison_chart(stock_data, valid_stocks)
        st.plotly_chart(volume_chart, use_container_width=True)

    # Export comparison data
    st.markdown("---")
    st.subheader("Export Comparison Data")

    # Prepare export data
    export_df = normalized_data.copy()
    export_df.index.name = 'Date'

    csv = export_df.to_csv()

    st.download_button(
        label="Download Comparison Data as CSV",
        data=csv,
        file_name=f"stock_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
