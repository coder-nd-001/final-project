import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="CryptoSage AI",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: #0a0a0f;
    }

    /* ---- HERO ---- */
    .hero-wrap {
        background: linear-gradient(135deg, #0d0d1a 0%, #0a0f2e 50%, #0d0d1a 100%);
        border: 1px solid #1a1a3e;
        border-radius: 20px;
        padding: 48px 40px 36px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: '';
        position: absolute;
        top: -80px; right: -80px;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-wrap::after {
        content: '';
        position: absolute;
        bottom: -60px; left: -60px;
        width: 240px; height: 240px;
        background: radial-gradient(circle, rgba(245,158,11,0.10) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.4);
        color: #818cf8;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 100px;
        margin-bottom: 16px;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.1;
        margin: 0 0 10px;
        letter-spacing: -0.03em;
    }
    .hero-title span {
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        color: #64748b;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
    }
    .hero-stats {
        display: flex;
        gap: 32px;
        margin-top: 28px;
        flex-wrap: wrap;
    }
    .hero-stat {
        display: flex;
        flex-direction: column;
    }
    .hero-stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 600;
        color: #f1f5f9;
    }
    .hero-stat-lbl {
        font-size: 0.72rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 2px;
    }
    .hero-divider {
        width: 1px;
        background: #1e293b;
        align-self: stretch;
    }

    /* ---- SECTION HEADER ---- */
    .sec-head {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 32px 0 16px;
    }
    .sec-head-icon {
        width: 32px; height: 32px;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
    }
    .sec-head-text {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        letter-spacing: -0.01em;
    }

    /* ---- KPI CARDS ---- */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #0f0f1a;
        border: 1px solid #1e1e3a;
        border-radius: 14px;
        padding: 20px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s;
    }
    .kpi-card:hover { border-color: #3730a3; }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--accent, linear-gradient(90deg, #6366f1, #818cf8));
    }
    .kpi-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #475569;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.45rem;
        font-weight: 600;
        color: #f1f5f9;
        line-height: 1.1;
    }
    .kpi-change {
        font-size: 0.78rem;
        margin-top: 6px;
        font-weight: 500;
    }
    .kpi-change.up   { color: #34d399; }
    .kpi-change.down { color: #f87171; }
    .kpi-change.neu  { color: #94a3b8; }

    /* ---- SIGNAL BANNER ---- */
    .signal-banner {
        border-radius: 14px;
        padding: 20px 24px;
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 24px;
        border: 1px solid;
    }
    .signal-banner.buy  { background: rgba(52,211,153,0.08); border-color: rgba(52,211,153,0.3); }
    .signal-banner.sell { background: rgba(248,113,113,0.08); border-color: rgba(248,113,113,0.3); }
    .signal-banner.hold { background: rgba(251,191,36,0.08);  border-color: rgba(251,191,36,0.3); }
    .signal-icon { font-size: 2rem; }
    .signal-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 2px; }
    .signal-title.buy  { color: #34d399; }
    .signal-title.sell { color: #f87171; }
    .signal-title.hold { color: #fbbf24; }
    .signal-desc  { font-size: 0.85rem; color: #64748b; }

    /* ---- RISK BADGE ---- */
    .risk-row { display: flex; gap: 10px; align-items: center; margin-bottom: 8px; }
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 100px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .risk-low  { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
    .risk-med  { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
    .risk-high { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

    /* ---- MODEL COMPARE ---- */
    .model-compare {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 24px;
    }
    .model-card {
        background: #0f0f1a;
        border: 1px solid #1e1e3a;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .model-card.active {
        border-color: #6366f1;
        background: rgba(99,102,241,0.07);
    }
    .model-name { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
    .model-pred { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 600; color: #f1f5f9; }
    .model-tag  { font-size: 0.7rem; margin-top: 6px; }
    .model-tag.active-tag { color: #818cf8; }
    .model-tag.inactive-tag { color: #334155; }

    /* ---- INSIGHT PILL ---- */
    .insight-pill {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 0.88rem;
        color: #94a3b8;
        display: flex;
        align-items: flex-start;
        gap: 10px;
    }
    .insight-pill .icon { color: #818cf8; font-size: 1rem; flex-shrink: 0; margin-top: 1px; }

    /* ---- FOOTER ---- */
    .footer {
        text-align: center;
        color: #1e293b;
        font-size: 0.78rem;
        padding: 24px 0 8px;
        border-top: 1px solid #0f172a;
        margin-top: 40px;
    }
    .footer span { color: #334155; }

    /* ---- SIDEBAR TWEAKS ---- */
    section[data-testid="stSidebar"] {
        background: #070710 !important;
        border-right: 1px solid #0f0f2a;
    }
    section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stRadio label { color: #64748b !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.06em; }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("### ₿ CryptoSage AI")
    st.markdown("---")

    coin = st.selectbox(
        "Cryptocurrency",
        ["bitcoin", "ethereum", "litecoin", "solana", "cardano", "dogecoin"],
        format_func=lambda x: {
            "bitcoin": "₿ Bitcoin (BTC)",
            "ethereum": "Ξ Ethereum (ETH)",
            "litecoin": "Ł Litecoin (LTC)",
            "solana": "◎ Solana (SOL)",
            "cardano": "₳ Cardano (ADA)",
            "dogecoin": "Ð Dogecoin (DOGE)"
        }[x]
    )

    st.markdown("---")

    investment = st.number_input(
        "Investment Amount (₹)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000
    )

    days = st.slider("Historical Data (Days)", 30, 365, 180)

    st.markdown("---")

    model_choice = st.radio(
        "Prediction Model",
        ["LSTM", "Linear Regression", "Hybrid (LSTM + LR)"],
        index=2
    )

    st.markdown("---")

    show_bollinger   = st.toggle("Bollinger Bands",    value=True)
    show_rsi         = st.toggle("RSI Indicator",      value=True)
    show_macd        = st.toggle("MACD Chart",         value=True)
    show_volume_dist = st.toggle("Volume Distribution", value=True)
    show_model_cmp   = st.toggle("Model Comparison",   value=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#1e293b;text-align:center'>CryptoSage AI v2.0<br>Powered by CoinGecko + TensorFlow</div>",
        unsafe_allow_html=True
    )

    page = st.sidebar.radio(
    "📊 Navigation",
    [
        "Dashboard",
        "Market Analysis",
        "Prediction Center",
        "Portfolio",
        "Risk Analytics",
        "Model Performance"
    ]
)
    st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "Data Source",
    [
        "CoinGecko Live",
        "Manual Dataset"
    ]
)

uploaded_file = None

if data_source == "Manual Dataset":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


# ================= FETCH DATA =================
COIN_SYMBOLS = {
    "bitcoin": "BTC", "ethereum": "ETH", "litecoin": "LTC",
    "solana": "SOL", "cardano": "ADA", "dogecoin": "DOGE"
}
symbol = COIN_SYMBOLS[coin]

@st.cache_data(ttl=300)
def fetch_crypto_data(coin, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    r = requests.get(url, params={"vs_currency": "inr", "days": days})
    data = r.json()
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # Volume (may not always exist)
    if "total_volumes" in data:
        vol_df = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
        vol_df["timestamp"] = pd.to_datetime(vol_df["timestamp"], unit="ms")
        vol_df.set_index("timestamp", inplace=True)
        df = df.join(vol_df, how="left")
    else:
        df["volume"] = np.nan

    # Market cap
    if "market_caps" in data:
        mc_df = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
        mc_df["timestamp"] = pd.to_datetime(mc_df["timestamp"], unit="ms")
        mc_df.set_index("timestamp", inplace=True)
        df = df.join(mc_df, how="left")

    return df

@st.cache_data(ttl=300)
def fetch_coin_info(coin):
    try:
        r = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin}",
            params={"localization": "false", "tickers": "false",
                    "market_data": "true", "community_data": "false"}
        )
        return r.json()
    except:
        return {}

with st.spinner("Fetching live market data…"):
    df = fetch_crypto_data(coin, days)
    info = fetch_coin_info(coin)

current_price = df["price"].iloc[-1]
price_7d_ago  = df["price"].iloc[-min(len(df), 200)] if len(df) >= 200 else df["price"].iloc[0]
change_7d     = ((current_price - price_7d_ago) / price_7d_ago) * 100
price_1d_ago  = df["price"].iloc[-min(len(df), 24)]
change_1d     = ((current_price - price_1d_ago) / price_1d_ago) * 100
high_period   = df["price"].max()
low_period    = df["price"].min()
avg_volume    = df["volume"].mean() if "volume" in df.columns else 0


# if data_source == "Manual Dataset" and uploaded_file:

#     df = pd.read_csv(uploaded_file)

#     df.columns = [
#         c.lower() for c in df.columns
#     ]

#     if "price" not in df.columns:
#         st.error(
#             "CSV must contain price column"
#         )
#         st.stop()

#     if "timestamp" in df.columns:
#         df["timestamp"] = pd.to_datetime(
#             df["timestamp"]
#         )
#         df.set_index(
#             "timestamp",
#             inplace=True
#         )

# else:
#     df = fetch_crypto_data(
#         coin,
#         days
#     )

# ================= TECHNICAL INDICATORS =================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger(series, window=20, num_std=2):
    sma    = series.rolling(window).mean()
    std    = series.rolling(window).std()
    upper  = sma + num_std * std
    lower  = sma - num_std * std
    return upper, sma, lower

df["rsi"]  = compute_rsi(df["price"])
df["ma20"] = df["price"].rolling(20).mean()
df["ma50"] = df["price"].rolling(50).mean()
df["bb_upper"], df["bb_mid"], df["bb_lower"] = compute_bollinger(df["price"])
df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["price"])

rsi_current = df["rsi"].iloc[-1]


# ================= HERO =================
market_cap_str = "—"
if "market_cap" in df.columns:
    mc = df["market_cap"].iloc[-1]
    if mc >= 1e12:
        market_cap_str = f"₹{mc/1e12:.2f}T"
    elif mc >= 1e9:
        market_cap_str = f"₹{mc/1e9:.2f}B"
    elif mc >= 1e6:
        market_cap_str = f"₹{mc/1e6:.2f}M"

rsi_label = "Overbought" if rsi_current > 70 else ("Oversold" if rsi_current < 30 else "Neutral")
chg1d_sign = "▲" if change_1d >= 0 else "▼"
chg1d_color = "#34d399" if change_1d >= 0 else "#f87171"

st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-badge">Live Market Intelligence</div>
    <p class="hero-title">{symbol} / INR &nbsp;<span>CryptoSage</span></p>
    <p class="hero-sub">Real-time price analysis · AI-powered forecasting · Risk assessment</p>
    <div class="hero-stats">
        <div class="hero-stat">
            <span class="hero-stat-val" style="font-family:'JetBrains Mono',monospace">₹{current_price:,.2f}</span>
            <span class="hero-stat-lbl">Current Price</span>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <span class="hero-stat-val" style="color:{chg1d_color}">{chg1d_sign} {abs(change_1d):.2f}%</span>
            <span class="hero-stat-lbl">24h Change</span>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <span class="hero-stat-val">{rsi_current:.1f}</span>
            <span class="hero-stat-lbl">RSI · {rsi_label}</span>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <span class="hero-stat-val">{market_cap_str}</span>
            <span class="hero-stat-lbl">Market Cap</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ================= KPI CARDS =================
st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)

def kpi(label, value, change=None, accent="linear-gradient(90deg,#6366f1,#818cf8)"):
    chg_html = ""
    if change is not None:
        cls  = "up" if change > 0 else ("down" if change < 0 else "neu")
        sign = "▲" if change > 0 else ("▼" if change < 0 else "—")
        chg_html = f'<div class="kpi-change {cls}">{sign} {abs(change):.2f}%</div>'
    return f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {chg_html}
    </div>"""

cards_html = (
    kpi("Current Price", f"₹{current_price:,.2f}", change_1d,
        "linear-gradient(90deg,#6366f1,#818cf8)") +
    kpi(f"{days}d High", f"₹{high_period:,.2f}", accent="linear-gradient(90deg,#34d399,#6ee7b7)") +
    kpi(f"{days}d Low",  f"₹{low_period:,.2f}",  accent="linear-gradient(90deg,#f87171,#fca5a5)") +
    kpi("7d Change", f"{change_7d:+.2f}%", change_7d,
        "linear-gradient(90deg,#f59e0b,#fbbf24)") +
    kpi("RSI (14)", f"{rsi_current:.1f}", accent="linear-gradient(90deg,#a78bfa,#c4b5fd)")
)
st.markdown(cards_html + '</div>', unsafe_allow_html=True)


# ================= MAIN PRICE CHART =================
st.markdown('<div class="sec-head"><div class="sec-head-icon">📈</div><div class="sec-head-text">Price Chart</div></div>', unsafe_allow_html=True)

price_fig = go.Figure()

if show_bollinger:
    price_fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_upper"], mode="lines",
        line=dict(color="rgba(99,102,241,0.3)", width=1, dash="dot"),
        name="BB Upper", showlegend=True
    ))
    price_fig.add_trace(go.Scatter(
        x=df.index, y=df["bb_lower"], mode="lines",
        line=dict(color="rgba(99,102,241,0.3)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(99,102,241,0.04)",
        name="BB Band"
    ))

price_fig.add_trace(go.Scatter(
    x=df.index, y=df["price"],
    mode="lines",
    line=dict(color="#f59e0b", width=2.5),
    name=f"{symbol} Price",
    hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra></extra>"
))
price_fig.add_trace(go.Scatter(
    x=df.index, y=df["ma20"], mode="lines",
    line=dict(color="#818cf8", width=1.2, dash="dash"),
    name="MA 20"
))
price_fig.add_trace(go.Scatter(
    x=df.index, y=df["ma50"], mode="lines",
    line=dict(color="#34d399", width=1.2, dash="dash"),
    name="MA 50"
))

price_fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,20,0.6)",
    height=420,
    font=dict(family="Space Grotesk", color="#64748b", size=12),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        bgcolor="rgba(0,0,0,0)", font=dict(size=11)
    ),
    xaxis=dict(gridcolor="#0f172a", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#0f172a", showgrid=True, zeroline=False,
               tickprefix="₹", tickformat=",.0f"),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=30, b=10)
)
st.plotly_chart(price_fig, use_container_width=True)


# ================= RSI + MACD =================
if show_rsi or show_macd:
    n_rows  = int(show_rsi) + int(show_macd)
    row_heights = []
    subplot_titles = []
    if show_rsi:
        row_heights.append(1); subplot_titles.append("RSI (14)")
    if show_macd:
        row_heights.append(1); subplot_titles.append("MACD")

    ind_fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08
    )
    cur_row = 1

    if show_rsi:
        ind_fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"],
            line=dict(color="#a78bfa", width=1.8),
            name="RSI", hovertemplate="RSI: %{y:.1f}<extra></extra>"
        ), row=cur_row, col=1)
        ind_fig.add_hline(y=70, line=dict(color="#f87171", dash="dot", width=1), row=cur_row, col=1)
        ind_fig.add_hline(y=30, line=dict(color="#34d399", dash="dot", width=1), row=cur_row, col=1)
        ind_fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,113,113,0.04)", line_width=0, row=cur_row, col=1)
        ind_fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(52,211,153,0.04)",  line_width=0, row=cur_row, col=1)
        cur_row += 1

    if show_macd:
        colors_hist = ["#34d399" if v >= 0 else "#f87171" for v in df["macd_hist"].fillna(0)]
        ind_fig.add_trace(go.Bar(
            x=df.index, y=df["macd_hist"],
            marker_color=colors_hist, name="Histogram", opacity=0.6
        ), row=cur_row, col=1)
        ind_fig.add_trace(go.Scatter(
            x=df.index, y=df["macd"],
            line=dict(color="#6366f1", width=1.5), name="MACD"
        ), row=cur_row, col=1)
        ind_fig.add_trace(go.Scatter(
            x=df.index, y=df["macd_signal"],
            line=dict(color="#f59e0b", width=1.5, dash="dash"), name="Signal"
        ), row=cur_row, col=1)

    ind_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.6)",
        height=200 * n_rows,
        font=dict(family="Space Grotesk", color="#64748b", size=11),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor="#0f172a"), yaxis=dict(gridcolor="#0f172a")
    )
    st.plotly_chart(ind_fig, use_container_width=True)


# ================= VOLUME DISTRIBUTION =================
if show_volume_dist and "volume" in df.columns and not df["volume"].isna().all():
    st.markdown('<div class="sec-head"><div class="sec-head-icon">📦</div><div class="sec-head-text">Volume Distribution</div></div>', unsafe_allow_html=True)
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(
        x=df.index, y=df["volume"],
        marker=dict(
            color=df["price"].pct_change().fillna(0).apply(
                lambda x: "rgba(52,211,153,0.5)" if x >= 0 else "rgba(248,113,113,0.5)"
            )
        ),
        name="Volume",
        hovertemplate="<b>%{x|%d %b}</b><br>Vol: %{y:,.0f}<extra></extra>"
    ))
    vol_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.6)",
        height=200,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#0f172a"),
        yaxis=dict(gridcolor="#0f172a", tickformat=".2s")
    )
    st.plotly_chart(vol_fig, use_container_width=True)


# ================= ML PREPROCESSING =================
prices_arr = df["price"].values.reshape(-1, 1)
scaler     = MinMaxScaler()
scaled     = scaler.fit_transform(prices_arr)

window_size = 15
X, y_arr = [], []
for i in range(window_size, len(scaled)):
    X.append(scaled[i - window_size:i])
    y_arr.append(scaled[i])

X     = np.array(X)
y_arr = np.array(y_arr)

split   = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_arr[:split], y_arr[split:]

# Linear Regression
X_lr_train = X_train.reshape(X_train.shape[0], -1)
X_lr_test  = X_test.reshape(X_test.shape[0], -1)
lr_model = LinearRegression()
lr_model.fit(X_lr_train, y_train)
lr_preds_scaled = lr_model.predict(X_lr_test).flatten()
lr_preds = scaler.inverse_transform(lr_preds_scaled.reshape(-1, 1)).flatten()

# LSTM
lstm_model = None
if model_choice in ["LSTM", "Hybrid (LSTM + LR)"]:
    with st.spinner("Training LSTM model…"):
        lstm_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16),
            Dense(1)
        ])
        lstm_model.compile(optimizer="adam", loss="huber")
        lstm_model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)
    lstm_preds_scaled = lstm_model.predict(X_test, verbose=0).flatten()
    lstm_preds = scaler.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()


# ================= FUTURE PREDICTION =================
last_window = scaled[-window_size:]
X_pred      = np.array([last_window])
X_pred_lr   = X_pred.reshape(1, -1)

lr_next_scaled   = lr_model.predict(X_pred_lr)[0][0]
lr_next          = scaler.inverse_transform([[lr_next_scaled]])[0][0]

if lstm_model is not None:
    lstm_next_scaled = lstm_model.predict(X_pred, verbose=0)[0][0]
    lstm_next        = scaler.inverse_transform([[lstm_next_scaled]])[0][0]
else:
    lstm_next_scaled = lr_next_scaled
    lstm_next        = lr_next

if model_choice == "Linear Regression":
    predicted_price = lr_next
elif model_choice == "LSTM":
    predicted_price = lstm_next
else:
    predicted_price = (lr_next + lstm_next) / 2

change_percent = ((predicted_price - current_price) / current_price) * 100


# ================= MODEL ACCURACY =================
y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

lr_mae  = mean_absolute_error(y_true, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_true, lr_preds))
lr_mape = np.mean(np.abs((y_true - lr_preds) / y_true)) * 100
lr_acc  = max(0, 100 - lr_mape)

if model_choice in ["LSTM", "Hybrid (LSTM + LR)"]:
    lstm_mae  = mean_absolute_error(y_true, lstm_preds)
    lstm_rmse = np.sqrt(mean_squared_error(y_true, lstm_preds))
    lstm_mape = np.mean(np.abs((y_true - lstm_preds) / y_true)) * 100
    lstm_acc  = max(0, 100 - lstm_mape)
else:
    lstm_mae = lstm_rmse = lstm_mape = lstm_acc = None


# ================= MODEL COMPARISON =================
if show_model_cmp:
    st.markdown('<div class="sec-head"><div class="sec-head-icon">🤖</div><div class="sec-head-text">Model Comparison</div></div>', unsafe_allow_html=True)

    hybrid_next = (lr_next + lstm_next) / 2 if lstm_model else lr_next

    is_lr     = model_choice == "Linear Regression"
    is_lstm   = model_choice == "LSTM"
    is_hybrid = model_choice == "Hybrid (LSTM + LR)"

    st.markdown(f"""
    <div class="model-compare">
        <div class="model-card {'active' if is_lr else ''}">
            <div class="model-name">Linear Regression</div>
            <div class="model-pred">₹{lr_next:,.2f}</div>
            <div class="model-tag {'active-tag' if is_lr else 'inactive-tag'}">
                {'✦ Active' if is_lr else f'Acc {lr_acc:.1f}%'}
            </div>
        </div>
        <div class="model-card {'active' if is_lstm else ''}">
            <div class="model-name">LSTM Neural Net</div>
            <div class="model-pred">₹{lstm_next:,.2f}</div>
            <div class="model-tag {'active-tag' if is_lstm else 'inactive-tag'}">
                {'✦ Active' if is_lstm else (f'Acc {lstm_acc:.1f}%' if lstm_acc else '—')}
            </div>
        </div>
        <div class="model-card {'active' if is_hybrid else ''}">
            <div class="model-name">Hybrid Ensemble</div>
            <div class="model-pred">₹{hybrid_next:,.2f}</div>
            <div class="model-tag {'active-tag' if is_hybrid else 'inactive-tag'}">
                {'✦ Active' if is_hybrid else 'Avg of both'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Backtest chart: actual vs predicted
    st.markdown('<div class="sec-head"><div class="sec-head-icon">🔬</div><div class="sec-head-text">Backtest — Actual vs Predicted</div></div>', unsafe_allow_html=True)

    bt_dates = df.index[window_size + split:]

    bt_fig = go.Figure()
    bt_fig.add_trace(go.Scatter(
        x=bt_dates, y=y_true,
        line=dict(color="#f59e0b", width=2), name="Actual"
    ))
    bt_fig.add_trace(go.Scatter(
        x=bt_dates, y=lr_preds,
        line=dict(color="#6366f1", width=1.5, dash="dash"), name="LR Prediction"
    ))
    if lstm_model is not None:
        bt_fig.add_trace(go.Scatter(
            x=bt_dates, y=lstm_preds,
            line=dict(color="#34d399", width=1.5, dash="dot"), name="LSTM Prediction"
        ))
    bt_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.6)",
        height=300,
        font=dict(family="Space Grotesk", color="#64748b", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#0f172a"),
        yaxis=dict(gridcolor="#0f172a", tickprefix="₹", tickformat=",.0f"),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(bt_fig, use_container_width=True)

    # Accuracy metrics
    cols = st.columns(4)
    metrics = [
        ("LR MAE",  f"₹{lr_mae:,.0f}"),
        ("LR RMSE", f"₹{lr_rmse:,.0f}"),
        ("LR Accuracy", f"{lr_acc:.1f}%"),
    ]
    if lstm_acc is not None:
        metrics.append(("LSTM Accuracy", f"{lstm_acc:.1f}%"))
    for col, (lbl, val) in zip(cols, metrics):
        col.metric(lbl, val)


# ================= PREDICTION RESULT =================
st.markdown('<div class="sec-head"><div class="sec-head-icon">🔮</div><div class="sec-head-text">Next Candle Prediction</div></div>', unsafe_allow_html=True)

if predicted_price > current_price * 1.02:
    sig, sig_class, sig_icon, sig_desc = (
        "BUY", "buy", "📈",
        f"{symbol} shows upward momentum. Model projects a {change_percent:.2f}% gain. Consider entering a position."
    )
elif predicted_price < current_price * 0.98:
    sig, sig_class, sig_icon, sig_desc = (
        "SELL", "sell", "📉",
        f"{symbol} shows downward pressure. Model projects a {abs(change_percent):.2f}% decline. Risk management advised."
    )
else:
    sig, sig_class, sig_icon, sig_desc = (
        "HOLD", "hold", "⚖️",
        f"{symbol} is in a consolidation zone. Price movement is within ±2%. Wait for a clearer signal."
    )

st.markdown(f"""
<div class="signal-banner {sig_class}">
    <div class="signal-icon">{sig_icon}</div>
    <div>
        <div class="signal-title {sig_class}">{sig} Signal — {model_choice}</div>
        <div class="signal-desc">{sig_desc}</div>
    </div>
    <div style="margin-left:auto;text-align:right">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;color:#f1f5f9;font-weight:600">₹{predicted_price:,.2f}</div>
        <div style="font-size:0.75rem;color:#475569">Predicted Price</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Prediction KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price",  f"₹{current_price:,.2f}")
c2.metric("Predicted Price", f"₹{predicted_price:,.2f}")
c3.metric("Expected Change", f"{change_percent:+.2f}%")
c4.metric("Active Model", model_choice.split()[0])


# ================= PORTFOLIO SIMULATION =================
st.markdown('<div class="sec-head"><div class="sec-head-icon">💼</div><div class="sec-head-text">Portfolio Simulation</div></div>', unsafe_allow_html=True)

coins_bought = investment / current_price
future_value = coins_bought * predicted_price
profit       = future_value - investment
roi          = (profit / investment) * 100

p1, p2, p3, p4 = st.columns(4)
p1.metric("Investment",   f"₹{investment:,.2f}")
p2.metric("Coins Bought", f"{coins_bought:.6f} {symbol}")
p3.metric("Future Value", f"₹{future_value:,.2f}")
p4.metric("P&L",          f"₹{profit:,.2f}", f"{roi:+.2f}%")

# Scenario table
st.markdown("**Scenario Analysis**")
scenarios = pd.DataFrame({
    "Scenario":       ["Bear (-10%)", "Base (Model)", "Bull (+10%)", "Bull (+20%)"],
    "Price":          [
        current_price * 0.90, predicted_price,
        current_price * 1.10, current_price * 1.20
    ],
    "Portfolio Value": [
        coins_bought * current_price * 0.90,
        future_value,
        coins_bought * current_price * 1.10,
        coins_bought * current_price * 1.20
    ],
    "P&L": [
        coins_bought * current_price * 0.90 - investment,
        profit,
        coins_bought * current_price * 1.10 - investment,
        coins_bought * current_price * 1.20 - investment
    ]
})
scenarios["Price"]            = scenarios["Price"].apply(lambda x: f"₹{x:,.2f}")
scenarios["Portfolio Value"]  = scenarios["Portfolio Value"].apply(lambda x: f"₹{x:,.2f}")
scenarios["P&L"]              = scenarios["P&L"].apply(lambda x: f"₹{x:+,.2f}")
st.dataframe(scenarios, use_container_width=True, hide_index=True)


# ================= RISK ANALYSIS =================
st.markdown('<div class="sec-head"><div class="sec-head-icon">⚠️</div><div class="sec-head-text">Risk Analysis</div></div>', unsafe_allow_html=True)

volatility     = df["price"].pct_change().std()
annualized_vol = volatility * np.sqrt(365) * 100
daily_returns  = df["price"].pct_change().dropna()
sharpe         = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() > 0 else 0
max_drawdown   = ((df["price"] / df["price"].cummax()) - 1).min() * 100

if volatility < 0.02:
    risk_level, risk_class = "LOW", "risk-low"
elif volatility < 0.05:
    risk_level, risk_class = "MEDIUM", "risk-med"
else:
    risk_level, risk_class = "HIGH", "risk-high"

st.markdown(f"""
<div class="risk-row">
    <span style="color:#64748b;font-size:0.85rem">Overall Risk:</span>
    <span class="risk-badge {risk_class}">● {risk_level} RISK</span>
</div>
""", unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Daily Volatility",   f"{volatility*100:.2f}%")
r2.metric("Annual Volatility",  f"{annualized_vol:.1f}%")
r3.metric("Sharpe Ratio",       f"{sharpe:.2f}")
r4.metric("Max Drawdown",       f"{max_drawdown:.1f}%")

# Return distribution
ret_fig = go.Figure()
ret_fig.add_trace(go.Histogram(
    x=daily_returns * 100,
    nbinsx=50,
    marker_color="#6366f1",
    opacity=0.7,
    name="Daily Returns"
))
ret_fig.add_vline(x=0, line=dict(color="#f59e0b", width=1.5, dash="dash"))
ret_fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,20,0.6)",
    height=240,
    title=dict(text="Daily Return Distribution", font=dict(size=13, color="#64748b")),
    xaxis=dict(title="Return %", gridcolor="#0f172a"),
    yaxis=dict(title="Frequency", gridcolor="#0f172a"),
    showlegend=False,
    margin=dict(l=10, r=10, t=40, b=10)
)
st.plotly_chart(ret_fig, use_container_width=True)


# ================= AI INSIGHTS =================
st.markdown('<div class="sec-head"><div class="sec-head-icon">📌</div><div class="sec-head-text">AI Market Insights</div></div>', unsafe_allow_html=True)

bb_pos = "near upper band (overbought zone)" if df["price"].iloc[-1] >= df["bb_upper"].iloc[-1] * 0.98 else \
         "near lower band (oversold zone)"   if df["price"].iloc[-1] <= df["bb_lower"].iloc[-1] * 1.02 else \
         "within Bollinger Bands (neutral range)"

macd_cross = "MACD line is above signal — bullish momentum" \
    if df["macd"].iloc[-1] > df["macd_signal"].iloc[-1] \
    else "MACD line is below signal — bearish momentum"

ma_trend = "above both MA20 and MA50 — strong uptrend" \
    if current_price > df["ma20"].iloc[-1] and current_price > df["ma50"].iloc[-1] \
    else "below both moving averages — downtrend" \
    if current_price < df["ma20"].iloc[-1] and current_price < df["ma50"].iloc[-1] \
    else "between MA20 and MA50 — mixed trend"

insights = [
    ("📊", f"<b>Trend:</b> {symbol} is {ma_trend}. This suggests {"accumulation" if "uptrend" in ma_trend else "distribution"} by market participants."),
    ("📉", f"<b>Momentum (RSI {rsi_current:.1f}):</b> {rsi_label} territory. {'Watch for potential reversal.' if rsi_current > 65 or rsi_current < 35 else 'No extreme momentum signal currently.'}"),
    ("🎯", f"<b>Bollinger Bands:</b> Price is {bb_pos}. This {'often precedes a pullback.' if 'upper' in bb_pos else 'may indicate a bounce.' if 'lower' in bb_pos else 'suggests consolidation.'}"),
    ("⚡", f"<b>MACD:</b> {macd_cross}. Histogram bars {'expanding' if abs(df['macd_hist'].iloc[-1]) > abs(df['macd_hist'].iloc[-5]) else 'compressing'} — momentum is {'strengthening' if abs(df['macd_hist'].iloc[-1]) > abs(df['macd_hist'].iloc[-5]) else 'weakening'}."),
    ("🛡️", f"<b>Risk:</b> {risk_level} risk with {annualized_vol:.1f}% annualised volatility. Sharpe ratio of {sharpe:.2f} {'indicates risk-adjusted returns are acceptable.' if sharpe > 1 else 'indicates caution is warranted.'}"),
]

for icon, text in insights:
    st.markdown(f'<div class="insight-pill"><span class="icon">{icon}</span><span>{text}</span></div>', unsafe_allow_html=True)


# ================= FOOTER =================
st.markdown(f"""
<div class="footer">
    Built by <span>Nagesh</span> · CryptoSage AI v2.0 ·
    Data: CoinGecko · Model: {model_choice} ·
    <span>Not financial advice</span>
</div>
""", unsafe_allow_html=True)
