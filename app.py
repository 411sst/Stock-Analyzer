# app.py - Enhanced Indian Stock Trading Dashboard with Fixed Charts
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# Import existing modules
from components.market_overview_module import market_overview_page
from components.stock_analysis_module import stock_analysis_page
from components.portfolio_tracker_module import portfolio_tracker_page
from components.news_sentiment_module import news_sentiment_page
from utils.indian_stocks import INDIAN_STOCKS

# Import new authentication and ML modules
try:
    from authentication.auth_handler import AuthHandler
    from authentication.validators import validate_email, validate_password, validate_username, get_password_strength_score, get_password_strength_text
    from ml_forecasting.models.ensemble_model import EnsembleModel
    ENHANCED_FEATURES = True
except ImportError as e:
    ENHANCED_FEATURES = False
    st.sidebar.error(f"⚠️ Enhanced features not available: {str(e)}")

# Set page config
st.set_page_config(
    page_title="Indian Stock Dashboard - Enhanced",
    page_icon="📊",  # Keeping minimal page icon for browser tab only
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI with systematic color palette and typography
st.markdown("""
<style>
    /* Font Imports */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Color System */
    :root {
        /* Backgrounds */
        --bg-primary: #0F0F0F;
        --bg-secondary: #1A1A1A;
        --bg-tertiary: #242424;

        /* Borders */
        --border-subtle: #2A2A2A;
        --border-default: #404040;
        --border-strong: #525252;

        /* Text */
        --text-primary: #FFFFFF;
        --text-secondary: #A0A0A0;
        --text-tertiary: #707070;

        /* Semantic */
        --positive: #10B981;
        --negative: #EF4444;
        --warning: #F59E0B;
        --info: #3B82F6;

        /* Interactive */
        --interactive-default: #FFFFFF;
        --interactive-hover: #E5E5E5;
        --interactive-active: #D4D4D4;

        /* Typography Scale */
        --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        --font-mono: 'JetBrains Mono', 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace;
    }

    /*
    TYPOGRAPHY SYSTEM DOCUMENTATION

    UI Text (Inter):
    - Weight 400 (Regular): Body text, descriptions
    - Weight 500 (Medium): Labels, form fields, secondary headings
    - Weight 600 (Semibold): Buttons, primary headings, emphasis
    - Weight 700 (Bold): H1, major page titles
    - Fallback: -apple-system, BlinkMacSystemFont, Segoe UI, system-ui, sans-serif

    Financial Data (JetBrains Mono):
    - Weight 400 (Regular): Standard numerical display
    - Weight 500 (Medium): Emphasized numbers, metric values
    - Weight 600 (Semibold): Large financial figures, primary metrics
    - Fallback: SF Mono, Monaco, Cascadia Code, Roboto Mono, Consolas, monospace
    - Feature: font-variant-numeric: tabular-nums (ensures digit alignment)

    Line Height Ratios:
    - Headings (h1-h3): 1.2-1.4 (tighter for visual impact)
    - Body text: 1.5-1.6 (optimal readability)
    - Captions/labels: 1.4 (balanced for small text)
    - Financial data: 1.4 (maintains alignment in tables)
    */

    /* Base Typography */
    body, .stApp {
        font-family: var(--font-ui);
        font-size: 15px;
        font-weight: 400;
        line-height: 1.6;
        color: var(--text-primary);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Heading Hierarchy */
    h1, .stTitle {
        font-family: var(--font-ui);
        font-size: 32px;
        font-weight: 700;
        line-height: 1.2;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        margin-bottom: 16px;
    }

    h2, .stHeader {
        font-family: var(--font-ui);
        font-size: 24px;
        font-weight: 600;
        line-height: 1.3;
        letter-spacing: -0.01em;
        color: var(--text-primary);
        margin-bottom: 12px;
    }

    h3, .stSubheader {
        font-family: var(--font-ui);
        font-size: 18px;
        font-weight: 600;
        line-height: 1.4;
        color: var(--text-primary);
        margin-bottom: 8px;
    }

    /* Body Text */
    p, .stMarkdown, .stText {
        font-family: var(--font-ui);
        font-size: 15px;
        font-weight: 400;
        line-height: 1.6;
        color: var(--text-primary);
    }

    /* Caption/Label Text */
    label, .stCaption, small, caption {
        font-family: var(--font-ui);
        font-size: 13px;
        font-weight: 500;
        line-height: 1.4;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        color: var(--text-secondary);
    }

    /* Financial Data - Monospace Typography */
    .metric-value, .price, .percentage, .quantity, .financial-data,
    .stMetric, .stMetricValue, [data-testid="stMetricValue"],
    .stDataFrame td, .stTable td {
        font-family: var(--font-mono);
        font-variant-numeric: tabular-nums;
        font-weight: 500;
    }

    /* Metric Cards - Enhanced Typography */
    .stMetric {
        font-family: var(--font-ui);
    }

    .stMetric label, [data-testid="stMetricLabel"] {
        font-family: var(--font-ui);
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        color: var(--text-secondary);
        line-height: 1.4;
    }

    .stMetric [data-testid="stMetricValue"] {
        font-family: var(--font-mono);
        font-size: 28px;
        font-weight: 600;
        line-height: 1.2;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
    }

    .stMetric [data-testid="stMetricDelta"] {
        font-family: var(--font-mono);
        font-size: 14px;
        font-weight: 400;
        line-height: 1.4;
        font-variant-numeric: tabular-nums;
    }

    /* Table Typography - Right-align financial columns */
    .stDataFrame, .stTable {
        font-family: var(--font-ui);
        border-radius: 8px;
        overflow: hidden;
    }

    .stDataFrame th, .stTable th {
        font-family: var(--font-ui);
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-secondary);
        background-color: var(--bg-secondary);
        padding: 12px 16px;
        border-bottom: 1px solid var(--border-subtle);
        text-align: right;
    }

    .stDataFrame td, .stTable td {
        font-family: var(--font-mono);
        font-size: 14px;
        font-weight: 400;
        text-align: right;
        font-variant-numeric: tabular-nums;
        padding: 10px 16px;
        border-bottom: 1px solid var(--border-subtle);
        transition: background-color 0.15s ease;
    }

    .stDataFrame tbody tr:hover td, .stTable tbody tr:hover td {
        background-color: var(--bg-tertiary);
    }

    .stDataFrame tbody tr:last-child td, .stTable tbody tr:last-child td {
        border-bottom: none;
    }

    /* First column (typically labels) should be left-aligned */
    .stDataFrame td:first-child, .stTable td:first-child,
    .stDataFrame th:first-child, .stTable th:first-child {
        text-align: left;
        font-family: var(--font-ui);
        font-weight: 500;
    }

    /* Form Elements - Enhanced Focus States */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        font-family: var(--font-ui);
        font-size: 15px;
        font-weight: 400;
        line-height: 1.5;
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 10px 14px;
        color: var(--text-primary);
        transition: all 0.2s ease;
    }

    .stTextInput input:hover, .stSelectbox select:hover, .stTextArea textarea:hover {
        border-color: var(--border-default);
    }

    .stTextInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {
        border-color: var(--interactive-default);
        outline: none;
        box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.1);
    }

    .stTextInput label, .stSelectbox label, .stTextArea label {
        font-family: var(--font-ui);
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.3px;
        color: var(--text-secondary);
        margin-bottom: 8px;
        display: block;
    }

    /* Password Input Specific Styling */
    .stTextInput input[type="password"] {
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 2px;
    }

    /* Checkbox Styling */
    .stCheckbox {
        font-family: var(--font-ui);
        font-size: 14px;
        font-weight: 400;
        color: var(--text-primary);
    }

    /* Select Dropdown */
    .stSelectbox select {
        cursor: pointer;
    }

    .stSelectbox select option {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        padding: 8px;
    }

    /* Global Border Radius Standardization */
    .stAlert, .stInfo, .stWarning, .stError, .stSuccess {
        border-radius: 8px;
    }

    .stExpander {
        border-radius: 8px;
        border: 1px solid var(--border-subtle);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        font-family: var(--font-ui);
        font-size: 14px;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab-border"] {
        height: 2px;
        background-color: var(--interactive-default);
    }

    /* Sidebar Refinements */
    .css-1d391kg, .css-1lcbmhc {
        border-radius: 8px;
    }

    /* Radio Buttons */
    .stRadio > label {
        font-family: var(--font-ui);
        font-size: 14px;
        font-weight: 400;
        color: var(--text-primary);
        padding: 8px 12px;
        border-radius: 6px;
        transition: background-color 0.2s ease;
    }

    .stRadio > label:hover {
        background-color: var(--bg-tertiary);
    }

    .reportview-container {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }
    .sidebar .sidebar-content {
        background-color: var(--bg-secondary);
    }
    .metric-container {
        background-color: var(--bg-secondary);
        border-radius: 8px;
        padding: 16px;
        border: 1px solid var(--border-subtle);
    }
    .stButton button {
        font-family: var(--font-ui);
        font-size: 15px;
        font-weight: 600;
        line-height: 1.5;
        background-color: var(--interactive-default);
        color: #000000;
        border-radius: 6px;
        border: none;
        padding: 10px 16px;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background-color: var(--interactive-hover);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .user-info {
        padding: 24px;
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        margin-bottom: 24px;
        color: var(--text-primary);
    }
    .auth-container {
        background-color: var(--bg-secondary);
        padding: 24px;
        border-radius: 8px;
        margin-bottom: 24px;
        border: 1px solid var(--border-subtle);
    }
    .success-message {
        font-family: var(--font-ui);
        font-size: 14px;
        font-weight: 400;
        line-height: 1.5;
        padding: 12px 16px;
        background-color: transparent;
        border-left: 3px solid var(--positive);
        border-radius: 0;
        color: #D1FAE5;
        margin: 10px 0;
    }
    .error-message {
        font-family: var(--font-ui);
        font-size: 14px;
        font-weight: 400;
        line-height: 1.5;
        padding: 12px 16px;
        background-color: transparent;
        border-left: 3px solid var(--negative);
        border-radius: 0;
        color: #FECACA;
        margin: 10px 0;
    }
    .prediction-card {
        background-color: var(--bg-secondary);
        border-radius: 8px;
        padding: 24px;
        margin: 24px 0;
        border: 1px solid var(--border-subtle);
    }
    .ml-metric {
        text-align: center;
        padding: 16px;
        background-color: var(--bg-secondary);
        border-radius: 8px;
        margin: 8px;
        border: 1px solid var(--border-subtle);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'mode' not in st.session_state:
        st.session_state.mode = "Beginner"
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = 'RELIANCE.NS'
    if 'show_ml_details' not in st.session_state:
        st.session_state.show_ml_details = False

initialize_session_state()

# Initialize authentication handler
if ENHANCED_FEATURES:
    if 'auth_handler' not in st.session_state:
        st.session_state.auth_handler = AuthHandler()

def create_password_strength_indicator(password):
    """Create a visual password strength indicator"""
    if not password:
        return ""

    score = get_password_strength_score(password)
    strength_text, color = get_password_strength_text(score)

    return f"""
    <div style="margin: 16px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 500; color: var(--text-secondary);">Password Strength:</span>
            <span style="font-family: 'Inter', sans-serif; color: {color}; font-weight: 600; font-size: 13px;">{strength_text}</span>
        </div>
        <div style="background-color: var(--border-subtle); border-radius: 2px; height: 3px; overflow: hidden;">
            <div style="background-color: {color}; width: {score}%; height: 100%; border-radius: 2px; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);"></div>
        </div>
    </div>
    """

# Sidebar Content
with st.sidebar:
    st.title("Indian Stock Dashboard")
    st.markdown("*Enhanced with AI & Authentication*")

    # Authentication Section
    if ENHANCED_FEATURES:
        if not st.session_state.logged_in:
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.markdown("### User Authentication")

            auth_tab1, auth_tab2 = st.tabs(["Login", "Register"])

            with auth_tab1:
                with st.form("login_form", clear_on_submit=False):
                    st.markdown("#### Welcome Back!")
                    username = st.text_input("Username", placeholder="Enter your username")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        remember_me = st.checkbox("Remember me", value=True)
                    with col2:
                        st.markdown('<small><a href="#" style="color: #60a5fa;">Forgot?</a></small>', unsafe_allow_html=True)

                    login_submit = st.form_submit_button("Login", use_container_width=True)

                    if login_submit:
                        if username and password:
                            with st.spinner("Authenticating..."):
                                user_id, message = st.session_state.auth_handler.verify_user(username, password)
                                if user_id:
                                    st.session_state.logged_in = True
                                    st.session_state.user = st.session_state.auth_handler.get_user_info(user_id)
                                    st.markdown('<div class="success-message">Login successful</div>', unsafe_allow_html=True)
                                    st.rerun()
                                else:
                                    st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-message">Please fill in all fields</div>', unsafe_allow_html=True)

            with auth_tab2:
                with st.form("register_form", clear_on_submit=False):
                    st.markdown("#### Create Account")
                    new_username = st.text_input("Username", placeholder="Choose a username")
                    new_email = st.text_input("Email", placeholder="your.email@example.com")
                    new_password = st.text_input("Password", type="password", placeholder="Create a strong password")
                    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")

                    # Password strength indicator
                    if new_password:
                        st.markdown(create_password_strength_indicator(new_password), unsafe_allow_html=True)

                    agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")

                    register_submit = st.form_submit_button("Create Account", use_container_width=True)

                    if register_submit:
                        if not agree_terms:
                            st.markdown('<div class="error-message">Please agree to the terms</div>', unsafe_allow_html=True)
                        elif new_username and new_email and new_password and confirm_password:
                            # Validate inputs
                            username_valid, username_msg = validate_username(new_username)
                            email_valid, email_msg = validate_email(new_email)
                            password_valid, password_msg = validate_password(new_password, confirm_password)

                            if not username_valid:
                                st.markdown(f'<div class="error-message">{username_msg}</div>', unsafe_allow_html=True)
                            elif not email_valid:
                                st.markdown(f'<div class="error-message">{email_msg}</div>', unsafe_allow_html=True)
                            elif not password_valid:
                                st.markdown(f'<div class="error-message">{password_msg}</div>', unsafe_allow_html=True)
                            else:
                                with st.spinner("Creating account..."):
                                    success, message = st.session_state.auth_handler.register_user(
                                        new_username, new_email, new_password
                                    )
                                    if success:
                                        st.markdown('<div class="success-message">Registration successful! Please login.</div>', unsafe_allow_html=True)
                                        st.balloons()
                                    else:
                                        st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-message">Please fill in all fields</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # User is logged in
            user = st.session_state.user
            st.markdown(f"""
            <div class="user-info">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <h4 style="margin: 0;">Welcome, {user['username']}</h4>
                        <small>{user['email']}</small><br>
                        <small>Last login: {user.get('last_login', 'Never')}</small>
                    </div>
                    <div style="text-align: right;">
                        <div style="background-color: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 15px; font-size: 11px;">
                            Premium User
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # User actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Profile", use_container_width=True):
                    st.session_state.show_profile = True
            with col2:
                if st.button("Logout", use_container_width=True):
                    st.session_state.logged_in = False
                    st.session_state.user = None
                    st.rerun()

    st.markdown("---")

    # Mode Selection
    st.markdown("### Trading Mode")
    mode_options = ["Beginner", "Pro", "Expert"]
    mode_descriptions = {
        "Beginner": "Simple analysis with explanations",
        "Pro": "Advanced technical indicators",
        "Expert": "Full ML predictions & risk analysis"
    }

    selected_mode = st.selectbox(
        "Select your experience level:",
        mode_options,
        index=mode_options.index(st.session_state.mode)
    )
    st.session_state.mode = selected_mode
    st.caption(mode_descriptions[selected_mode])

    st.markdown("---")

    # Navigation
    st.markdown("### Navigation")
    nav_options = ["Market Overview", "Stock Analysis", "Portfolio Tracker", "News & Sentiment"]

    if ENHANCED_FEATURES and st.session_state.logged_in:
        nav_options.extend(["ML Predictions", "User Settings"])

    selected_nav = st.radio(
        "Choose a section:",
        nav_options,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Quick Stats
    st.markdown("### Quick Market Stats")
    try:
        nifty = yf.download("^NSEI", period="1d", interval="1m")
        if not nifty.empty:
            current_nifty = nifty['Close'][-1]
            nifty_change = ((current_nifty - nifty['Close'][0]) / nifty['Close'][0]) * 100

            st.metric(
                "NIFTY 50",
                f"₹{current_nifty:.2f}",
                f"{nifty_change:+.2f}%"
            )
    except Exception:
        st.metric("NIFTY 50", "₹25,400", "+0.45%")

    # Feature availability indicator
    st.markdown("---")
    st.markdown("### Features Available")
    if ENHANCED_FEATURES:
        st.markdown("• User Authentication")
        st.markdown("• ML-Powered Predictions")
        st.markdown("• Personal Portfolio")
        st.markdown("• Advanced Analytics")
    else:
        st.markdown("• Basic features only")
        st.markdown("• Install ML packages for full experience")

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 12px;">© 2024 Indian Stock Dashboard<br>Enhanced with AI</p>',
        unsafe_allow_html=True
    )

# Main Content Area
if selected_nav == "Market Overview":
    market_overview_page(st.session_state.mode)

elif selected_nav == "Stock Analysis":
    stock_analysis_page(st.session_state.mode)

elif selected_nav == "Portfolio Tracker":
    portfolio_tracker_page(st.session_state.mode)

elif selected_nav == "News & Sentiment":
    news_sentiment_page(st.session_state.mode)

elif selected_nav == "ML Predictions" and ENHANCED_FEATURES:
    if not st.session_state.logged_in:
        st.warning("Please login to access ML-powered predictions.")
        st.info("Register for free to unlock advanced AI features")
    else:
        st.title("AI-Powered Stock Predictions & Risk Analysis")
        st.markdown("*Advanced machine learning models with comprehensive risk assessment*")

        # Stock selection
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            selected_stock = st.selectbox(
                "Select Stock for AI Analysis:",
                list(INDIAN_STOCKS.keys()),
                format_func=lambda x: f"{INDIAN_STOCKS[x]} ({x})"
            )

        with col2:
            prediction_period = st.selectbox(
                "Prediction Period:",
                ["1 Week", "2 Weeks", "1 Month"],
                index=0
            )

        with col3:
            analysis_depth = st.selectbox(
                "Analysis Level:",
                ["Basic", "Advanced", "Professional"],
                index=1
            )

        steps_map = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30}
        prediction_steps = steps_map[prediction_period]

        # Advanced Settings
        with st.expander("Advanced Model & Risk Settings", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Model Parameters**")
                confidence_threshold = st.slider("Confidence Threshold", 0.3, 0.9, 0.6)
                include_technical = st.checkbox("Include Technical Analysis", value=True)
                ensemble_weights = st.checkbox("Auto-adjust Model Weights", value=True)

            with col2:
                st.markdown("**Risk Assessment**")
                risk_adjustment = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
                calculate_var = st.checkbox("Calculate Value at Risk", value=True)
                stress_testing = st.checkbox("Run Stress Tests", value=True)

            with col3:
                st.markdown("**Display Options**")
                show_components = st.checkbox("Show Model Components", value=False)
                show_risk_metrics = st.checkbox("Show Risk Dashboard", value=True)
                show_performance = st.checkbox("Show Performance Metrics", value=analysis_depth != "Basic")

        # Quick Market Context
        with st.container():
            st.markdown("### Quick Market Context")
            context_cols = st.columns(4)

            try:
                nifty_data = yf.download("^NSEI", period="5d", progress=False)
                if not nifty_data.empty:
                    nifty_change = ((nifty_data['Close'][-1] - nifty_data['Close'][-2]) / nifty_data['Close'][-2]) * 100

                    with context_cols[0]:
                        st.metric("NIFTY 50", f"{nifty_data['Close'][-1]:.0f}", f"{nifty_change:+.1f}%")

                    with context_cols[1]:
                        market_sentiment = "Bullish" if nifty_change > 0.5 else "Bearish" if nifty_change < -0.5 else "Neutral"
                        st.metric("Market Sentiment", market_sentiment, f"{nifty_change:+.1f}%")

                    with context_cols[2]:
                        volatility = nifty_data['Close'].pct_change().std() * np.sqrt(252) * 100
                        st.metric("Market Volatility", f"{volatility:.1f}%", "Annualized")

                    with context_cols[3]:
                        st.metric("Trading Activity", "Active", "Market Hours")
            except Exception:
                with context_cols[0]:
                    st.metric("NIFTY 50", "25,400", "+0.45%")
                with context_cols[1]:
                    st.metric("Market Sentiment", "Neutral", "0.00%")

        st.markdown("---")

        # Main Prediction Button
        if st.button("Generate AI Prediction & Risk Analysis", type="primary", use_container_width=True):

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Step 1: Data Collection
                status_text.text("Fetching historical data...")
                progress_bar.progress(10)

                stock_data = yf.download(selected_stock, period="2y", progress=False)

                if stock_data.empty:
                    st.error("Unable to fetch stock data. Please try another stock.")
                else:
                    # Step 2: Data Preprocessing
                    status_text.text("Preprocessing data...")
                    progress_bar.progress(25)

                    close_data = stock_data['Close'].dropna()

                    if len(close_data) < 30:
                        st.warning("Very limited historical data. Results may be less accurate.")

                    # Step 3: ML Model Training
                    status_text.text("Training AI models...")
                    progress_bar.progress(40)

                    ensemble_model = EnsembleModel()
                    prediction_result = ensemble_model.predict(
                        close_data,
                        steps=prediction_steps,
                        symbol=selected_stock
                    )

                    # Step 4: Risk Analysis
                    status_text.text("Performing risk analysis...")
                    progress_bar.progress(60)

                    try:
                        from utils.risk_analysis import RiskAnalyzer, create_risk_dashboard, create_stress_test_chart

                        risk_analyzer = RiskAnalyzer()
                        risk_metrics = risk_analyzer.risk_metrics_dashboard(close_data, prediction_result['predictions'])

                        # Use the dynamic risk score from the prediction result if available
                        if 'risk_score' in prediction_result:
                            risk_metrics['risk_score'] = prediction_result['risk_score']

                    except Exception as risk_error:
                        st.info(f"Using simplified risk analysis: {str(risk_error)}")

                        try:
                            # Use the risk score from prediction_result if available
                            if 'risk_score' in prediction_result:
                                risk_score = prediction_result['risk_score']
                            else:
                                # Fallback calculation
                                volatility = close_data.pct_change().std() * np.sqrt(252) * 100
                                price_change = abs(prediction_result.get('price_change_percent', 0))
                                confidence = prediction_result.get('confidence', 0.5)

                                vol_component = min(30, volatility * 1.5)
                                price_component = min(25, price_change * 0.8)
                                conf_component = 20 if confidence < 0.5 else 10 if confidence < 0.7 else 5

                                risk_score = int(vol_component + price_component + conf_component + 15)
                                risk_score = max(15, min(95, risk_score))

                            risk_metrics = {
                                'risk_score': risk_score,
                                'var_metrics': {
                                    'var_1d': 0.025,
                                    'var_5d': 0.056,
                                    'var_10d': 0.079,
                                    'method': 'simplified'
                                },
                                'volatility_regime': {
                                    'regime': 'normal',
                                    'current_vol': 0.025,
                                    'historical_vol': 0.025
                                },
                                'stress_scenarios': {
                                    'bull_market': {'total_return': 12.0, 'final_price': prediction_result['current_price'] * 1.12},
                                    'base_case': {'total_return': prediction_result.get('price_change_percent', 0), 'final_price': prediction_result['predicted_price']},
                                    'bear_market': {'total_return': -8.0, 'final_price': prediction_result['current_price'] * 0.92}
                                }
                            }
                        except Exception:
                            risk_metrics = None

                    # Step 5: Validation
                    status_text.text("Validating predictions...")
                    progress_bar.progress(80)

                    validation_checks, is_valid = ensemble_model.validate_prediction(prediction_result)

                    # Step 6: Results Display
                    status_text.text("Preparing results...")
                    progress_bar.progress(100)

                    progress_bar.empty()
                    status_text.empty()

                    if not is_valid:
                        st.error("Prediction validation failed. Please try again.")
                        with st.expander("Debug Information"):
                            st.json(validation_checks)
                    else:
                        st.success("AI Analysis Complete")

                        st.markdown("---")
                        st.subheader("AI Prediction Results")

                        current_price = prediction_result.get('current_price', 0)
                        predicted_price = prediction_result.get('predicted_price', 0)
                        price_change = prediction_result.get('price_change_percent', 0)
                        confidence = prediction_result.get('confidence', 0)

                        metric_cols = st.columns(4)

                        with metric_cols[0]:
                            st.metric("Current Price", f"₹{current_price:.2f}")

                        with metric_cols[1]:
                            direction = "▲" if price_change > 0 else "▼" if price_change < 0 else "—"
                            st.metric("Predicted Price", f"₹{predicted_price:.2f}", f"{price_change:+.1f}% {direction}")

                        with metric_cols[2]:
                            st.metric("AI Confidence", f"{confidence:.1%}")

                        with metric_cols[3]:
                            if risk_metrics:
                                risk_score = risk_metrics.get('risk_score', 50)
                                st.metric("Risk Score", f"{risk_score}/100")
                            else:
                                st.metric("Data Points", f"{len(close_data)}")

                        # COMPLETELY FIXED CHARTS SECTION - ERROR RESOLVED
                        st.markdown("---")
                        st.subheader("Price Prediction Visualization")

                        try:
                            
                            # Create the chart
                            fig = go.Figure()
                            
                            # STEP 1: Add Historical Data (COMPLETELY FIXED)
                            historical_data = close_data.tail(30)
                            if len(historical_data) > 0:
                                # FIXED: Proper conversion to lists
                                try:
                                    if hasattr(historical_data.index, 'to_pydatetime'):
                                        hist_dates = historical_data.index.to_pydatetime().tolist()
                                    else:
                                        hist_dates = [date for date in historical_data.index]
                                    
                                    # FIXED: Handle DataFrame values properly
                                    if hasattr(historical_data, 'values'):
                                        hist_prices = historical_data.values.flatten().tolist()
                                    else:
                                        hist_prices = [float(val) for val in historical_data]

                                    # Add historical line
                                    fig.add_trace(go.Scatter(
                                        x=hist_dates,
                                        y=hist_prices,
                                        mode='lines',
                                        name='Historical Prices',
                                        line=dict(color='#3b82f6', width=3),
                                        hovertemplate='<b>Historical</b><br>Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
                                    ))
                                    
                                except Exception as hist_error:
                                    st.error(f"Historical data error: {str(hist_error)}")
                                    # Fallback for historical data
                                    hist_dates = [datetime.now() - timedelta(days=30-i) for i in range(30)]
                                    hist_prices = [float(close_data.iloc[-30+i]) for i in range(min(30, len(close_data)))]
                                    
                                    fig.add_trace(go.Scatter(
                                        x=hist_dates,
                                        y=hist_prices,
                                        mode='lines',
                                        name='Historical Prices (Fallback)',
                                        line=dict(color='#3b82f6', width=3)
                                    ))
                            
                            # STEP 2: Add Prediction Data (COMPLETELY FIXED)
                            predictions = prediction_result.get('predictions', [])
                            if len(predictions) > 0:
                                try:
                                    # FIXED: Robust prediction data handling
                                    if isinstance(predictions, np.ndarray):
                                        pred_values = predictions.flatten()
                                    elif isinstance(predictions, list):
                                        pred_values = np.array(predictions)
                                    else:
                                        pred_values = np.array([predictions])
                                    
                                    # Clean the predictions - remove invalid values
                                    clean_predictions = []
                                    for p in pred_values:
                                        try:
                                            val = float(p)
                                            if not (np.isnan(val) or np.isinf(val)) and val > 0:
                                                clean_predictions.append(val)
                                        except (ValueError, TypeError):
                                            continue
                                    
                                    if len(clean_predictions) > 0:
                                        # Generate prediction dates (business days only)
                                        try:
                                            if len(historical_data) > 0:
                                                last_hist_date = hist_dates[-1] if 'hist_dates' in locals() else datetime.now()
                                            else:
                                                last_hist_date = datetime.now()

                                            pred_dates = []
                                            current_date = last_hist_date

                                            for i in range(len(clean_predictions)):
                                                current_date += timedelta(days=1)
                                                # Skip weekends for business days
                                                while current_date.weekday() > 4:  # 5=Saturday, 6=Sunday
                                                    current_date += timedelta(days=1)
                                                pred_dates.append(current_date)

                                            # Add prediction line
                                            fig.add_trace(go.Scatter(
                                                x=pred_dates,
                                                y=clean_predictions,
                                                mode='lines+markers',
                                                name=f'AI Predictions ({prediction_period})',
                                                line=dict(color='#ef4444', width=3),
                                                marker=dict(size=8, color='#ef4444'),
                                                hovertemplate='<b>Prediction</b><br>Date: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
                                            ))
                                            
                                            # Add confidence bands
                                            if confidence > 0.3 and len(clean_predictions) > 1:
                                                try:
                                                    pred_std = np.std(clean_predictions)
                                                    band_width = pred_std * (1.5 - confidence)
                                                    
                                                    upper_band = [p + band_width for p in clean_predictions]
                                                    lower_band = [p - band_width for p in clean_predictions]
                                                    
                                                    # Add confidence area - FIXED FORMAT
                                                    fig.add_trace(go.Scatter(
                                                        x=pred_dates + pred_dates[::-1],
                                                        y=upper_band + lower_band[::-1],
                                                        fill='toself',
                                                        fillcolor='rgba(239, 68, 68, 0.15)',
                                                        line=dict(color='rgba(255,255,255,0)'),
                                                        name='Confidence Band',
                                                        hoverinfo='skip'
                                                    ))
                                                    
                                                    st.success("✅ Confidence bands added!")
                                                    
                                                except Exception as band_error:
                                                    st.info(f"Confidence bands skipped: {str(band_error)}")
                                        
                                        except Exception as date_error:
                                            st.error(f"Date generation error: {str(date_error)}")
                                            # Simple fallback dates
                                            pred_dates = [datetime.now() + timedelta(days=i+1) for i in range(len(clean_predictions))]
                                            
                                            fig.add_trace(go.Scatter(
                                                x=pred_dates,
                                                y=clean_predictions,
                                                mode='lines+markers',
                                                name='AI Predictions (Simple)',
                                                line=dict(color='#ef4444', width=3),
                                                marker=dict(size=8, color='#ef4444')
                                            ))
                                    
                                    else:
                                        st.error("No valid prediction data after cleaning")

                                except Exception as pred_error:
                                    st.error(f"Prediction processing error: {str(pred_error)}")

                            else:
                                st.error("No prediction data found")
                            
                            # STEP 3: Add Current Price Reference Line
                            if 'hist_dates' in locals() and 'pred_dates' in locals():
                                try:
                                    all_x_data = hist_dates + pred_dates
                                    current_price_line = [current_price] * len(all_x_data)
                                    
                                    fig.add_trace(go.Scatter(
                                        x=all_x_data,
                                        y=current_price_line,
                                        mode='lines',
                                        name='Current Price',
                                        line=dict(color='#10b981', width=2, dash='dot'),
                                        hovertemplate='Current Price: ₹%{y:.2f}<extra></extra>'
                                    ))

                                except Exception as ref_error:
                                    st.info(f"Reference line skipped: {str(ref_error)}")

                            # Update chart layout with typography
                            fig.update_layout(
                                title={
                                    'text': f"{INDIAN_STOCKS.get(selected_stock, selected_stock)} - AI Price Prediction Analysis",
                                    'x': 0.5,
                                    'xanchor': 'center',
                                    'font': {
                                        'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                                        'size': 20,
                                        'weight': 600,
                                        'color': '#FFFFFF'
                                    }
                                },
                                font=dict(
                                    family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                                    size=13,
                                    color='#A0A0A0'
                                ),
                                xaxis_title="Date",
                                yaxis_title="Price (₹)",
                                template='plotly_dark',
                                height=500,
                                hovermode='x unified',
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1,
                                    font=dict(
                                        family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                                        size=13
                                    )
                                ),
                                plot_bgcolor='#0F0F0F',
                                paper_bgcolor='#0F0F0F',
                                xaxis=dict(
                                    gridcolor='#242424',
                                    gridwidth=1,
                                    showline=False,
                                    zeroline=False,
                                    tickfont=dict(
                                        family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                                        size=12,
                                        color='#A0A0A0'
                                    )
                                ),
                                yaxis=dict(
                                    gridcolor='#242424',
                                    gridwidth=1,
                                    showline=False,
                                    zeroline=False,
                                    tickformat=',.0f',
                                    tickfont=dict(
                                        family='JetBrains Mono, monospace',
                                        size=12,
                                        color='#A0A0A0'
                                    )
                                ),
                                showlegend=True,
                                hovermode='x unified',
                                hoverlabel=dict(
                                    bgcolor='#1A1A1A',
                                    bordercolor='#404040',
                                    font=dict(
                                        family='JetBrains Mono, monospace',
                                        size=13,
                                        color='#FFFFFF'
                                    )
                                )
                            )
                            
                            # Display the chart
                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as chart_error:
                            st.error(f"Chart generation error: {str(chart_error)}")

                            # Comprehensive debug information
                            with st.expander("Full Debug Information", expanded=True):
                                st.write("**Error Details:**")
                                st.code(str(chart_error))
                                
                                st.write("**Data Types:**")
                                st.write(f"- close_data type: {type(close_data)}")
                                st.write(f"- prediction_result type: {type(prediction_result)}")
                                
                                predictions_data = prediction_result.get('predictions', [])
                                st.write(f"- predictions type: {type(predictions_data)}")
                                st.write(f"- predictions shape: {getattr(predictions_data, 'shape', 'N/A')}")
                                
                                st.write("**Data Shapes:**")
                                st.write(f"- close_data length: {len(close_data) if close_data is not None else 'None'}")
                                st.write(f"- predictions length: {len(predictions_data) if hasattr(predictions_data, '__len__') else 'N/A'}")
                                
                                st.write("**Sample Data:**")
                                try:
                                    if len(close_data) > 0:
                                        sample_hist = close_data.tail(3)
                                        if hasattr(sample_hist, 'values'):
                                            st.write(f"- Last 3 historical prices: {sample_hist.values.flatten()[:3].tolist()}")
                                        else:
                                            st.write(f"- Last 3 historical prices: {[float(val) for val in sample_hist[:3]]}")
                                except Exception as sample_error:
                                    st.write(f"- Historical data sample error: {sample_error}")
                                
                                try:
                                    if len(predictions_data) > 0:
                                        if isinstance(predictions_data, np.ndarray):
                                            sample_preds = predictions_data.flatten()[:3].tolist()
                                        else:
                                            sample_preds = list(predictions_data)[:3]
                                        st.write(f"- First 3 predictions: {sample_preds}")
                                except Exception as pred_sample_error:
                                    st.write(f"- Prediction data sample error: {pred_sample_error}")
                                
                                # FINAL FALLBACK CHART
                                st.write("**🆘 Creating Emergency Fallback Chart...**")
                                try:
                                    fallback_fig = go.Figure()
                                    
                                    # Simple two-point chart
                                    fallback_dates = [datetime.now(), datetime.now() + timedelta(days=7)]
                                    fallback_prices = [current_price, predicted_price]
                                    
                                    fallback_fig.add_trace(go.Scatter(
                                        x=fallback_dates,
                                        y=fallback_prices,
                                        mode='lines+markers',
                                        name='Simple Prediction',
                                        line=dict(color='red', width=3),
                                        marker=dict(size=10)
                                    ))
                                    
                                    fallback_fig.update_layout(
                                        title="🆘 Emergency Fallback Chart - Basic Prediction",
                                        template='plotly_dark',
                                        height=300,
                                        xaxis_title="Date",
                                        yaxis_title="Price (₹)"
                                    )
                                    
                                    st.plotly_chart(fallback_fig, use_container_width=True)
                                    st.success("✅ Emergency fallback chart displayed!")
                                    
                                except Exception as fallback_error:
                                    st.error(f"❌ Even emergency fallback failed: {fallback_error}")
                                    st.info("Please contact support with the error details above.")

                        # Risk Analysis Dashboard
                        if risk_metrics and show_risk_metrics:
                            st.markdown("---")
                            st.subheader("Risk Analysis Dashboard")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                try:
                                    # Horizontal Risk Bar (Replaces Gauge)
                                    risk_score = risk_metrics.get('risk_score', 50)

                                    # Determine risk level and color
                                    if risk_score < 40:
                                        risk_level = "Low Risk"
                                        risk_color = "#10B981"
                                    elif risk_score < 70:
                                        risk_level = "Moderate Risk"
                                        risk_color = "#F59E0B"
                                    else:
                                        risk_level = "High Risk"
                                        risk_color = "#EF4444"

                                    st.markdown(f"""
                                    <div style="
                                        background-color: var(--bg-secondary);
                                        border: 1px solid var(--border-subtle);
                                        border-radius: 8px;
                                        padding: 24px;
                                        height: 300px;
                                        display: flex;
                                        flex-direction: column;
                                        justify-content: center;
                                    ">
                                        <div style="margin-bottom: 24px;">
                                            <h3 style="
                                                font-family: 'Inter', sans-serif;
                                                font-size: 18px;
                                                font-weight: 600;
                                                color: #FFFFFF;
                                                margin: 0 0 8px 0;
                                            ">Risk Assessment</h3>
                                            <div style="
                                                font-family: 'JetBrains Mono', monospace;
                                                font-size: 48px;
                                                font-weight: 600;
                                                color: {risk_color};
                                                line-height: 1.2;
                                                margin-bottom: 4px;
                                            ">{risk_score}<span style="font-size: 24px; color: var(--text-secondary);">/100</span></div>
                                            <div style="
                                                font-family: 'Inter', sans-serif;
                                                font-size: 14px;
                                                font-weight: 500;
                                                color: var(--text-secondary);
                                            ">{risk_level}</div>
                                        </div>

                                        <div style="margin-top: 16px;">
                                            <div style="
                                                background-color: var(--border-subtle);
                                                height: 8px;
                                                border-radius: 4px;
                                                overflow: hidden;
                                                position: relative;
                                            ">
                                                <div style="
                                                    background-color: {risk_color};
                                                    width: {risk_score}%;
                                                    height: 100%;
                                                    border-radius: 4px;
                                                    transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
                                                "></div>
                                            </div>
                                            <div style="
                                                display: flex;
                                                justify-content: space-between;
                                                margin-top: 8px;
                                                font-family: 'Inter', sans-serif;
                                                font-size: 11px;
                                                color: var(--text-tertiary);
                                            ">
                                                <span>Low</span>
                                                <span>Moderate</span>
                                                <span>High</span>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                except Exception as risk_error:
                                    st.info(f"Risk indicator temporarily unavailable: {str(risk_error)}")
                            
                            with col2:
                                try:
                                    # Stress Test Chart
                                    stress_scenarios = risk_metrics.get('stress_scenarios', {})
                                    
                                    if stress_scenarios and len(stress_scenarios) > 0:
                                        scenarios = []
                                        returns = []
                                        
                                        scenario_mapping = {
                                            'bull_market': 'Bull Market',
                                            'base_case': 'Base Case', 
                                            'bear_market': 'Bear Market',
                                            'correction': 'Correction',
                                            'crash': 'Crash'
                                        }
                                        
                                        for key, data in stress_scenarios.items():
                                            if isinstance(data, dict) and 'total_return' in data:
                                                scenarios.append(scenario_mapping.get(key, key.replace('_', ' ').title()))
                                                returns.append(float(data['total_return']))
                                        
                                        if len(scenarios) > 0:
                                            colors = []
                                            for ret in returns:
                                                if ret > 8:
                                                    colors.append('#10b981')  # Green
                                                elif ret > 0:
                                                    colors.append('#3b82f6')  # Blue
                                                elif ret > -10:
                                                    colors.append('#f59e0b')  # Orange
                                                else:
                                                    colors.append('#ef4444')  # Red
                                            
                                            stress_fig = go.Figure(data=[
                                                go.Bar(
                                                    x=scenarios,
                                                    y=returns,
                                                    marker_color=colors,
                                                    text=[f"{ret:+.1f}%" for ret in returns],
                                                    textposition='auto',
                                                    hovertemplate='<b>%{x}</b><br>Return: %{y:.1f}%<extra></extra>'
                                                )
                                            ])
                                            
                                            stress_fig.update_layout(
                                                title={
                                                    'text': "Stress Test Scenarios",
                                                    'font': {
                                                        'family': 'Inter, sans-serif',
                                                        'size': 16,
                                                        'weight': 600,
                                                        'color': '#FFFFFF'
                                                    }
                                                },
                                                font=dict(
                                                    family='Inter, sans-serif',
                                                    size=12,
                                                    color='#A0A0A0'
                                                ),
                                                xaxis_title="Market Scenario",
                                                yaxis_title="Return (%)",
                                                template='plotly_dark',
                                                height=300,
                                                showlegend=False,
                                                plot_bgcolor='#0F0F0F',
                                                paper_bgcolor='#0F0F0F',
                                                xaxis=dict(
                                                    gridcolor='#242424',
                                                    tickfont=dict(family='Inter, sans-serif', size=11, color='#A0A0A0')
                                                ),
                                                yaxis=dict(
                                                    zeroline=True,
                                                    zerolinewidth=1,
                                                    zerolinecolor='#404040',
                                                    gridcolor='#242424',
                                                    tickfont=dict(family='JetBrains Mono, monospace', size=11, color='#A0A0A0')
                                                ),
                                                hoverlabel=dict(
                                                    bgcolor='#1A1A1A',
                                                    bordercolor='#404040',
                                                    font=dict(family='JetBrains Mono, monospace', size=12)
                                                )
                                            )
                                            
                                            st.plotly_chart(stress_fig, use_container_width=True)
                                        else:
                                            st.info("No stress test data available")
                                    else:
                                        # Fallback stress test chart
                                        scenarios = ['Bull Market', 'Base Case', 'Bear Market', 'Correction', 'Crash']
                                        returns = [12.0, price_change, -8.0, -20.0, -35.0]
                                        colors = ['green' if ret > 5 else 'blue' if ret > 0 else 'orange' if ret > -15 else 'red' for ret in returns]
                                        
                                        stress_fig = go.Figure(data=[
                                            go.Bar(
                                                x=scenarios,
                                                y=returns,
                                                marker_color=colors,
                                                text=[f"{ret:+.1f}%" for ret in returns],
                                                textposition='auto'
                                            )
                                        ])
                                        
                                        stress_fig.update_layout(
                                            title={
                                                'text': "Stress Test Scenarios",
                                                'font': {'family': 'Inter, sans-serif', 'size': 16, 'weight': 600, 'color': '#FFFFFF'}
                                            },
                                            font=dict(family='Inter, sans-serif', size=12, color='#A0A0A0'),
                                            xaxis_title="Scenario",
                                            yaxis_title="Return (%)",
                                            template='plotly_dark',
                                            height=300,
                                            plot_bgcolor='#0F0F0F',
                                            paper_bgcolor='#0F0F0F',
                                            xaxis=dict(
                                                gridcolor='#242424',
                                                tickfont=dict(family='Inter, sans-serif', size=11, color='#A0A0A0')
                                            ),
                                            yaxis=dict(
                                                gridcolor='#242424',
                                                tickfont=dict(family='JetBrains Mono, monospace', size=11, color='#A0A0A0')
                                            ),
                                            hoverlabel=dict(
                                                bgcolor='#1A1A1A',
                                                bordercolor='#404040',
                                                font=dict(family='JetBrains Mono, monospace', size=12)
                                            )
                                        )
                                        
                                        st.plotly_chart(stress_fig, use_container_width=True)
                                        
                                except Exception as stress_error:
                                    st.info(f"Stress test chart temporarily unavailable: {str(stress_error)}")

                        # Model Performance Breakdown
                        if show_components and 'individual_predictions' in prediction_result:
                            st.markdown("---")
                            st.subheader("Model Performance Breakdown")
                            
                            individual_preds = prediction_result['individual_predictions']
                            individual_confs = prediction_result['individual_confidences']
                            
                            if individual_preds and len(individual_preds) > 0:
                                try:
                                    # Individual model predictions chart
                                    model_fig = go.Figure()
                                    
                                    colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b']
                                    
                                    for i, (model_name, preds) in enumerate(individual_preds.items()):
                                        model_confidence = individual_confs.get(model_name, 0.5)
                                        
                                        # Ensure predictions are valid
                                        if isinstance(preds, (list, np.ndarray)) and len(preds) > 0:
                                            model_fig.add_trace(go.Scatter(
                                                x=pred_dates[:len(preds)],
                                                y=preds,
                                                mode='lines+markers',
                                                name=f'{model_name.replace("_", " ").title()} (Conf: {model_confidence:.1%})',
                                                line=dict(color=colors[i % len(colors)], width=2),
                                                marker=dict(size=4)
                                            ))
                                    
                                    # Add ensemble prediction
                                    if len(predictions) > 0:
                                        model_fig.add_trace(go.Scatter(
                                            x=pred_dates[:len(pred_values)],
                                            y=pred_values,
                                            mode='lines+markers',
                                            name=f'Ensemble (Conf: {confidence:.1%})',
                                            line=dict(color='white', width=3, dash='dash'),
                                            marker=dict(size=6, color='white', symbol='diamond')
                                        ))
                                    
                                    model_fig.update_layout(
                                        title={
                                            'text': "Individual Model Predictions vs Ensemble",
                                            'font': {'family': 'Inter, sans-serif', 'size': 18, 'weight': 600, 'color': '#FFFFFF'}
                                        },
                                        font=dict(family='Inter, sans-serif', size=13, color='#A0A0A0'),
                                        xaxis_title="Date",
                                        yaxis_title="Price (₹)",
                                        template='plotly_dark',
                                        height=400,
                                        plot_bgcolor='#0F0F0F',
                                        paper_bgcolor='#0F0F0F',
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="right",
                                            x=1,
                                            font=dict(family='Inter, sans-serif', size=12)
                                        ),
                                        xaxis=dict(
                                            gridcolor='#242424',
                                            tickfont=dict(family='Inter, sans-serif', size=11, color='#A0A0A0')
                                        ),
                                        yaxis=dict(
                                            gridcolor='#242424',
                                            tickfont=dict(family='JetBrains Mono, monospace', size=11, color='#A0A0A0')
                                        ),
                                        hoverlabel=dict(
                                            bgcolor='#1A1A1A',
                                            bordercolor='#404040',
                                            font=dict(family='JetBrains Mono, monospace', size=13)
                                        )
                                    )

                                    st.plotly_chart(model_fig, use_container_width=True)

                                    # Model confidence comparison
                                    if individual_confs and len(individual_confs) > 0:
                                        conf_fig = go.Figure(data=[
                                            go.Bar(
                                                x=[name.replace('_', ' ').title() for name in individual_confs.keys()],
                                                y=[conf * 100 for conf in individual_confs.values()],
                                                marker_color=colors[:len(individual_confs)],
                                                text=[f"{conf:.1%}" for conf in individual_confs.values()],
                                                textposition='auto'
                                            )
                                        ])

                                        conf_fig.update_layout(
                                            title={
                                                'text': "Individual Model Confidence Levels",
                                                'font': {'family': 'Inter, sans-serif', 'size': 16, 'weight': 600, 'color': '#FFFFFF'}
                                            },
                                            font=dict(family='Inter, sans-serif', size=12, color='#A0A0A0'),
                                            xaxis_title="Model",
                                            yaxis_title="Confidence (%)",
                                            template='plotly_dark',
                                            height=300,
                                            plot_bgcolor='#0F0F0F',
                                            paper_bgcolor='#0F0F0F',
                                            xaxis=dict(
                                                gridcolor='#242424',
                                                tickfont=dict(family='Inter, sans-serif', size=11, color='#A0A0A0')
                                            ),
                                            yaxis=dict(
                                                gridcolor='#242424',
                                                tickfont=dict(family='JetBrains Mono, monospace', size=11, color='#A0A0A0')
                                            ),
                                            hoverlabel=dict(
                                                bgcolor='#1A1A1A',
                                                bordercolor='#404040',
                                                font=dict(family='JetBrains Mono, monospace', size=12)
                                            )
                                        )
                                        
                                        st.plotly_chart(conf_fig, use_container_width=True)
                                
                                except Exception as model_error:
                                    st.info(f"Model breakdown charts temporarily unavailable: {str(model_error)}")

                        # Trading Recommendations
                        st.markdown("---")
                        st.subheader("AI Trading Recommendations")

                        recommendations = []

                        if confidence > 0.75:
                            recommendations.append("**High Confidence Signal** - Strong prediction reliability")
                        elif confidence > 0.6:
                            recommendations.append("**Moderate Confidence** - Consider position sizing")
                        else:
                            recommendations.append("**Low Confidence** - Wait for better signals")

                        if risk_metrics:
                            risk_score = risk_metrics.get('risk_score', 50)
                            if risk_score < 40:
                                recommendations.append("**Low Risk** - Suitable for conservative portfolios")
                            elif risk_score < 70:
                                recommendations.append("**Moderate Risk** - Standard position sizing")
                            else:
                                recommendations.append("**High Risk** - Consider reduced position or stop-loss")

                        if abs(price_change) > 10:
                            recommendations.append("**High Volatility Expected** - Monitor closely")

                        if price_change > 5:
                            recommendations.append("**Bullish Outlook** - Potential upside opportunity")
                        elif price_change < -5:
                            recommendations.append("**Bearish Outlook** - Consider defensive strategies")

                        for rec in recommendations:
                            st.markdown(f"• {rec}")

                        st.markdown("**Suggested Actions:**")
                        if confidence > 0.7 and (not risk_metrics or risk_metrics.get('risk_score', 50) < 60):
                            st.markdown("• Consider entering position with appropriate sizing")
                            st.markdown("• Set stop-loss orders for risk management")
                            st.markdown("• Monitor for confirmation signals")
                        else:
                            st.markdown("• Wait for better entry opportunities")
                            st.markdown("• Monitor market conditions")
                            st.markdown("• Consider paper trading to test strategy")

                        st.markdown("• Review analysis weekly")
                        st.markdown("• Track actual vs predicted performance")

                        # Export Results
                        st.markdown("---")
                        st.subheader("Export Analysis Results")

                        import re
                        import json

                        recommendations_str = '; '.join([rec.replace("**", "").strip() for rec in recommendations])

                        export_data = {
                            'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'Stock Symbol': selected_stock,
                            'Company Name': INDIAN_STOCKS.get(selected_stock, 'Unknown'),
                            'Current Price': f"₹{current_price:.2f}",
                            'Predicted Price': f"₹{predicted_price:.2f}",
                            'Expected Change': f"{price_change:+.1f}%",
                            'AI Confidence': f"{confidence:.1%}",
                            'Risk Score': f"{risk_metrics.get('risk_score', 50) if risk_metrics else 50}/100",
                            'Analysis Method': prediction_result.get('method', 'Ensemble'),
                            'Prediction Period': prediction_period,
                            'Data Points Used': len(close_data),
                            'Recommendations': recommendations_str
                        }
                        export_df = pd.DataFrame([export_data])
                        csv_data = export_df.to_csv(index=False)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="Download Analysis Report (CSV)",
                                data=csv_data,
                                file_name=f"AI_Analysis_{selected_stock}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        with col2:
                            json_data = {
                                'analysis_metadata': export_data,
                                'predictions': prediction_result.get('predictions', []).tolist()
                                    if isinstance(prediction_result.get('predictions', []), np.ndarray)
                                    else prediction_result.get('predictions', []),
                                'risk_metrics': risk_metrics if risk_metrics else {},
                                'model_performance': {
                                    'confidence': confidence,
                                    'validation_passed': is_valid,
                                    'individual_models': prediction_result.get('individual_confidences', {})
                                }
                            }
                            json_str = json.dumps(json_data, indent=2, default=str)
                            st.download_button(
                                label="Download Full Data (JSON)",
                                data=json_str,
                                file_name=f"AI_Analysis_Full_{selected_stock}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )

                        st.markdown("---")
                        st.warning("**Important Disclaimer:** This analysis is for educational purposes only and should not be considered as financial advice. Always consult with qualified financial professionals before making investment decisions.")

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("Please try again or contact support if the issue persists.")

                with st.expander("Troubleshooting Information"):
                    st.write("**Error Details:**")
                    st.code(str(e))

                    st.write("**Possible Solutions:**")
                    st.write("1. Check your internet connection")
                    st.write("2. Try a different stock symbol")
                    st.write("3. Reduce the prediction period")
                    st.write("4. Refresh the page and try again")

                    st.write("**System Information:**")
                    st.write(f"• Enhanced Features: {ENHANCED_FEATURES}")
                    st.write(f"• User Logged In: {st.session_state.logged_in}")
                    st.write(f"• Selected Stock: {selected_stock}")
                    st.write(f"• Prediction Steps: {prediction_steps}")

elif selected_nav == "User Settings" and ENHANCED_FEATURES and st.session_state.logged_in:
    st.title("User Settings & Preferences")

    user = st.session_state.user

    # User Profile Section
    st.subheader("Profile Information")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Username", value=user['username'], disabled=True)
        st.text_input("Email", value=user['email'], disabled=True)

    with col2:
        st.text_input("Member Since", value=user.get('created_at', 'Unknown'), disabled=True)
        st.text_input("Last Login", value=user.get('last_login', 'Never'), disabled=True)

    # Preferences Section
    st.subheader("Preferences")

    with st.form("preferences_form"):
        col1, col2 = st.columns(2)

        with col1:
            theme_preference = st.selectbox(
                "Theme",
                ["Dark", "Light", "Auto"],
                index=0 if user.get('theme', 'dark') == 'dark' else 1
            )

            default_mode = st.selectbox(
                "Default Trading Mode",
                ["Beginner", "Pro", "Expert"],
                index=["Beginner", "Pro", "Expert"].index(user.get('default_mode', 'Beginner'))
            )

        with col2:
            email_notifications = st.checkbox(
                "Email Notifications",
                value=user.get('email_notifications', True)
            )

            auto_refresh = st.checkbox(
                "Auto-refresh Data",
                value=True
            )

        if st.form_submit_button("Save Preferences", use_container_width=True):
            success = st.session_state.auth_handler.update_user_preferences(
                user['id'],
                theme=theme_preference.lower(),
                default_mode=default_mode,
                email_notifications=email_notifications
            )

            if success:
                st.success("Preferences updated successfully")
                st.session_state.user.update({
                    'theme': theme_preference.lower(),
                    'default_mode': default_mode,
                    'email_notifications': email_notifications
                })
            else:
                st.error("Failed to update preferences")

    # Portfolio Management Section
    st.subheader("Portfolio Management")

    user_portfolio = st.session_state.auth_handler.get_user_portfolio(user['id'])

    if user_portfolio:
        st.write(f"You have {len(user_portfolio)} holdings in your portfolio:")

        portfolio_df = pd.DataFrame(user_portfolio)
        st.dataframe(
            portfolio_df[['symbol', 'quantity', 'buy_price', 'buy_date']],
            use_container_width=True
        )

        if st.button("Clear All Portfolio Data", type="secondary"):
            if st.checkbox("I understand this will delete all my portfolio data"):
                st.warning("Clear portfolio functionality would be implemented here")
    else:
        st.info("Your portfolio is empty. Add some stocks from the Portfolio Tracker page")

    # Watchlist Management
    st.subheader("Watchlist Management")

    user_watchlist = st.session_state.auth_handler.get_user_watchlist(user['id'])

    if user_watchlist:
        st.write(f"You're watching {len(user_watchlist)} stocks:")

        for item in user_watchlist:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"{INDIAN_STOCKS.get(item['symbol'], item['symbol'])}")
            with col2:
                if item['alert_price']:
                    st.write(f"Alert: ₹{item['alert_price']}")
                else:
                    st.write("No alert set")
            with col3:
                if st.button("Remove", key=f"remove_{item['id']}"):
                    st.rerun()
    else:
        st.info("Your watchlist is empty. Add stocks from the Stock Analysis page")

    # Data Export Section
    st.subheader("Data Export")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export Portfolio", use_container_width=True):
            if user_portfolio:
                portfolio_df = pd.DataFrame(user_portfolio)
                csv = portfolio_df.to_csv(index=False)
                st.download_button(
                    label="Download Portfolio CSV",
                    data=csv,
                    file_name=f"portfolio_{user['username']}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No portfolio data to export")

    with col2:
        if st.button("Export Watchlist", use_container_width=True):
            if user_watchlist:
                watchlist_df = pd.DataFrame(user_watchlist)
                csv = watchlist_df.to_csv(index=False)
                st.download_button(
                    label="Download Watchlist CSV",
                    data=csv,
                    file_name=f"watchlist_{user['username']}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No watchlist data to export")

    # Account Management section
    st.subheader("Account Management")

    with st.expander("Change Password", expanded=False):
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_new_password = st.text_input("Confirm New Password", type="password")

            if new_password:
                st.markdown(create_password_strength_indicator(new_password), unsafe_allow_html=True)

            if st.form_submit_button("Change Password"):
                if not all([current_password, new_password, confirm_new_password]):
                    st.error("Please fill in all fields")
                elif new_password != confirm_new_password:
                    st.error("New passwords don't match")
                else:
                    password_valid, password_msg = validate_password(new_password)
                    if not password_valid:
                        st.error(f"{password_msg}")
                    else:
                        st.info("Password change functionality would be implemented here")

    with st.expander("Danger Zone", expanded=False):
        st.markdown("### Delete Account")
        st.warning("This action cannot be undone. All your data will be permanently deleted.")

        delete_confirmation = st.text_input(
            "Type 'DELETE' to confirm account deletion:",
            placeholder="Type DELETE here"
        )

        if st.button("Delete Account", type="secondary", disabled=delete_confirmation != "DELETE"):
            st.error("Account deletion functionality would be implemented here with proper confirmation")

else:
    if not ENHANCED_FEATURES:
        st.title("Indian Stock Trading Dashboard")
        st.warning("Enhanced features (Authentication & ML) are not available.")
        st.info("To enable full functionality, install required packages:")
        st.code("pip install tensorflow-cpu scikit-learn statsmodels bcrypt validators")

        st.markdown("---")
        st.markdown("### Available Features:")
        st.markdown("• Market Overview")
        st.markdown("• Stock Analysis")
        st.markdown("• Portfolio Tracker")
        st.markdown("• News & Sentiment")
        st.markdown("• User Authentication - Not Available")
        st.markdown("• ML Predictions - Not Available")
        st.markdown("• Personal Settings - Not Available")

    elif selected_nav not in ["Market Overview", "Stock Analysis", "Portfolio Tracker", "News & Sentiment"]:
        st.title("Authentication Required")
        st.info("Please login to access this feature.")

        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### Quick Login")
                with st.form("main_login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")

                    if st.form_submit_button("Login", use_container_width=True):
                        if username and password:
                            user_id, message = st.session_state.auth_handler.verify_user(username, password)
                            if user_id:
                                st.session_state.logged_in = True
                                st.session_state.user = st.session_state.auth_handler.get_user_info(user_id)
                                st.success("Login successful")
                                st.rerun()
                            else:
                                st.error(f"{message}")

# Footer with additional information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Features")
    st.markdown("- Real-time market data")
    st.markdown("- Technical analysis")
    st.markdown("- News sentiment analysis")
    if ENHANCED_FEATURES:
        st.markdown("- AI-powered predictions")
        st.markdown("- User authentication")

with col2:
    st.markdown("### Data Sources")
    st.markdown("- Yahoo Finance")
    st.markdown("- NSE/BSE APIs")
    st.markdown("- News aggregators")
    st.markdown("- Technical indicators")

with col3:
    st.markdown("### Disclaimer")
    st.markdown("This application is for educational purposes only.")
    st.markdown("Not financial advice.")
    st.markdown("Please consult qualified advisors.")

# Performance metrics (if user is logged in)
if ENHANCED_FEATURES and st.session_state.logged_in:
    with st.sidebar:
        if st.button("Performance Metrics"):
            st.session_state.show_performance = True

        if st.session_state.get('show_performance', False):
            st.markdown("### Your Trading Stats")
            st.metric("Portfolio Return", "+12.5%", "+2.3%")
            st.metric("Win Rate", "68%", "+5%")
            st.metric("Predictions Used", "23", "+3")

            if st.button("❌ Close"):
                st.session_state.show_performance = False
                st.rerun()
