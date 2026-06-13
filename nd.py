import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")

# =====================================================
#  PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="CryptoSage AI",
    page_icon="CS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
#  CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
    --bg: #080a0f;
    --panel: rgba(17, 20, 30, .92);
    --panel-2: rgba(22, 27, 39, .94);
    --panel-3: rgba(11, 14, 22, .96);
    --stroke: rgba(148, 163, 184, .16);
    --stroke-strong: rgba(148, 163, 184, .28);
    --text: #edf2f7;
    --muted: #9aa8ba;
    --faint: #627084;
    --brand: #f5b84b;
    --brand-soft: rgba(245, 184, 75, .14);
    --blue: #5aa7ff;
    --green: #22c55e;
    --red: #ef4444;
    --violet: #a78bfa;
    --shadow: 0 18px 50px rgba(0,0,0,.28);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background:
        linear-gradient(180deg, rgba(8,10,15,.96) 0%, rgba(10,13,20,.98) 48%, rgba(8,10,15,1) 100%),
        url("https://images.unsplash.com/photo-1642104704074-907c0698cbd9?auto=format&fit=crop&w=1800&q=75");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2.2rem 2.4rem !important; max-width: 1400px; }

/* ---- GLOBAL STREAMLIT SURFACES ---- */
div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--stroke);
    border-radius: 8px;
    padding: 16px 18px;
    box-shadow: 0 10px 28px rgba(0,0,0,.18);
}
div[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: .76rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
div[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'JetBrains Mono', monospace; font-size: 1.35rem !important; font-weight: 700; }
div[data-testid="stMetricDelta"] { font-weight: 700; }
.stDataFrame, div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; border: 1px solid var(--stroke); }
.stPlotlyChart { background: rgba(8,10,15,.45); border: 1px solid var(--stroke); border-radius: 8px; padding: 8px; }
hr { border-color: var(--stroke) !important; margin: 1rem 0 !important; }

/* ---- FORM CONTROLS ---- */
.stButton button, div[data-testid="stFormSubmitButton"] button {
    border-radius: 8px !important;
    border: 1px solid var(--stroke-strong) !important;
    background: linear-gradient(180deg, #1b2230, #121824) !important;
    color: var(--text) !important;
    font-weight: 700 !important;
    min-height: 40px;
    box-shadow: none !important;
    transition: border-color .15s ease, background .15s ease, transform .12s ease;
}
.stButton button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    border-color: rgba(245,184,75,.55) !important;
    background: linear-gradient(180deg, #232b3a, #151b28) !important;
    transform: translateY(-1px);
}
.stButton button[kind="primary"] {
    background: linear-gradient(180deg, #f6c35f, #d79525) !important;
    border-color: rgba(245,184,75,.8) !important;
    color: #15100a !important;
}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"], textarea {
    background: rgba(14, 18, 27, .9) !important;
    border-color: var(--stroke) !important;
    border-radius: 8px !important;
}
label, .stSlider label, .stRadio label, .stToggle label { color: var(--muted) !important; font-size: .76rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: .06em; }

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: rgba(7, 9, 14, .98) !important;
    border-right: 1px solid var(--stroke);
    box-shadow: 18px 0 50px rgba(0,0,0,.24);
}
section[data-testid="stSidebar"] * { color: var(--muted) !important; }
section[data-testid="stSidebar"] h3 { color: var(--text) !important; font-size: 1.05rem !important; font-weight: 800 !important; letter-spacing: 0 !important; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--faint) !important; font-size: .72rem !important; }

/* ---- NAV BUTTON ROW ---- */
div[data-testid="column"] > div:has(> div.stButton) button {
    border-radius: 8px !important;
    background: rgba(16, 20, 29, .82) !important;
    border: 1px solid var(--stroke) !important;
}

/* ---- HERO ---- */
.hero-wrap {
    background: linear-gradient(135deg, rgba(20,24,35,.96), rgba(13,16,24,.96));
    border: 1px solid var(--stroke);
    border-radius: 8px;
    padding: 30px 32px 26px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow);
}
.hero-wrap::before { content:''; position:absolute; left:0; top:0; bottom:0; width:4px; background: linear-gradient(180deg, var(--brand), var(--blue)); }
.hero-badge {
    display:inline-flex; align-items:center; gap:8px;
    background: var(--brand-soft);
    border: 1px solid rgba(245,184,75,.28);
    color: #f8d58a;
    font-size:.72rem; font-weight:800; letter-spacing:.08em;
    text-transform:uppercase; padding:5px 11px; border-radius:999px; margin-bottom:14px;
}
.hero-title { font-size:2.1rem; font-weight:800; color:var(--text); line-height:1.1; margin:0 0 8px; letter-spacing:0; }
.hero-title span { color: var(--brand); -webkit-text-fill-color: unset; background: none; }
.hero-sub { color:var(--muted); font-size:.96rem; margin:0; }
.hero-stats { display:flex; gap:20px; margin-top:22px; flex-wrap:wrap; }
.hero-stat { display:flex; flex-direction:column; min-width: 128px; }
.hero-stat-val { font-family:'JetBrains Mono',monospace; font-size:1.12rem; font-weight:700; color:var(--text); }
.hero-stat-lbl { font-size:.68rem; color:var(--faint); text-transform:uppercase; letter-spacing:.07em; margin-top:4px; font-weight:700; }
.hero-divider { width:1px; background:var(--stroke); align-self:stretch; }

/* ---- KPI CARDS ---- */
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(185px,1fr)); gap:12px; margin-bottom:20px; }
.kpi-card {
    background: var(--panel);
    border:1px solid var(--stroke);
    border-radius:8px;
    padding:17px 18px;
    position:relative;
    overflow:hidden;
    box-shadow: 0 10px 30px rgba(0,0,0,.16);
}
.kpi-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:var(--acc, var(--brand)); }
.kpi-label { font-size:.69rem; text-transform:uppercase; letter-spacing:.075em; color:var(--muted); font-weight:800; margin-bottom:8px; }
.kpi-value { font-family:'JetBrains Mono',monospace; font-size:1.18rem; font-weight:700; color:var(--text); line-height:1.25; overflow-wrap:anywhere; }
.kpi-ch { font-size:.76rem; margin-top:8px; font-weight:800; }
.kpi-ch.up{color:var(--green);} .kpi-ch.dn{color:var(--red);} .kpi-ch.ne{color:var(--muted);}

/* ---- SIGNALS & BADGES ---- */
.signal-banner, .ai-rec-box {
    border-radius:8px; padding:18px 20px; display:flex; align-items:center; gap:14px; margin-bottom:20px; border:1px solid var(--stroke); background: var(--panel);
    box-shadow: 0 12px 34px rgba(0,0,0,.18);
}
.signal-banner.buy, .ai-rec-box.buy { border-left:4px solid var(--green); background:rgba(16,30,24,.94); }
.signal-banner.sell, .ai-rec-box.sell { border-left:4px solid var(--red); background:rgba(32,18,21,.94); }
.signal-banner.hold, .ai-rec-box.hold { border-left:4px solid var(--brand); background:rgba(32,27,17,.94); }
.sig-icon, .ai-rec-icon { display:none; }
.sig-title, .ai-rec-title { font-size:1.05rem; font-weight:800; margin-bottom:3px; }
.sig-title.buy,.ai-rec-title.buy{color:var(--green);} .sig-title.sell,.ai-rec-title.sell{color:var(--red);} .sig-title.hold,.ai-rec-title.hold{color:var(--brand);}
.sig-desc, .ai-rec-desc { font-size:.84rem; color:var(--muted); }
.ai-rec-amount { margin-left:auto; text-align:right; }
.ai-rec-amount-val { font-family:'JetBrains Mono',monospace; font-size:1.12rem; font-weight:800; color:var(--text); }
.ai-rec-amount-lbl { font-size:.68rem; color:var(--faint); text-transform:uppercase; letter-spacing:.07em; font-weight:700; }

/* ---- SECTION HEADER ---- */
.sec-head { display:flex; align-items:center; gap:10px; margin:26px 0 13px; }
.sec-icon { width:28px;height:28px; background:rgba(245,184,75,.12); border:1px solid rgba(245,184,75,.24); border-radius:8px; display:flex;align-items:center;justify-content:center; font-size:.9rem; }
.sec-text { font-size:1rem; font-weight:800; color:var(--text); letter-spacing:0; }

/* ---- DASHBOARD / MODEL CARDS ---- */
.dash-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:20px; }
.dash-coin-card, .model-card, .mm-card, .trade-summary {
    background: var(--panel);
    border:1px solid var(--stroke);
    border-radius:8px;
    box-shadow: 0 12px 34px rgba(0,0,0,.18);
}
.dash-coin-card { padding:18px; position:relative; overflow:hidden; transition:border-color .15s ease, transform .12s ease; }
.dash-coin-card:hover { border-color:rgba(245,184,75,.45); transform:translateY(-1px); }
.dash-coin-top { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px; gap:12px; }
.dash-coin-name { font-size:.78rem; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; font-weight:800; }
.dash-coin-sym { font-size:.72rem; color:var(--faint); font-weight:700; }
.dash-coin-price { font-family:'JetBrains Mono',monospace; font-size:1.08rem; font-weight:800; color:var(--text); overflow-wrap:anywhere; }
.dash-coin-change { font-size:.76rem; font-weight:800; padding:4px 9px; border-radius:999px; white-space:nowrap; }
.dash-coin-change.up { background:rgba(34,197,94,.12); color:#5ee68a; }
.dash-coin-change.dn { background:rgba(239,68,68,.12); color:#ff7b7b; }

.model-compare { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:20px; }
.model-card { padding:16px; text-align:left; }
.model-card.active { border-color:rgba(245,184,75,.55); background:rgba(34,28,17,.94); }
.model-name { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; margin-bottom:8px; font-weight:800; }
.model-pred { font-family:'JetBrains Mono',monospace; font-size:1rem; font-weight:800; color:var(--text); overflow-wrap:anywhere; }
.model-tag { font-size:.72rem; margin-top:8px; font-weight:700; }
.model-tag.at{color:var(--brand);} .model-tag.it{color:var(--faint);}

/* ---- MANUAL MODEL PAGE ---- */
.mm-card { padding:22px; margin-bottom:16px; }
.mm-title { font-size:1rem; font-weight:800; color:var(--text); margin-bottom:4px; }
.mm-desc { font-size:.84rem; color:var(--muted); margin-bottom:16px; }
.mm-result-box { background:var(--panel-3); border:1px solid var(--stroke); border-radius:8px; padding:18px; text-align:center; }
.mm-result-label { font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px; font-weight:800; }
.mm-result-value { font-family:'JetBrains Mono',monospace; font-size:1.65rem; font-weight:800; color:var(--brand); }

/* ---- INVEST PANEL ---- */
.invest-btn-wrap button {
    background: linear-gradient(180deg, #f7c76a, #d69627) !important;
    color:#14100a !important;
    border:1px solid rgba(245,184,75,.8) !important;
    border-radius:8px !important;
    font-weight:900 !important;
    box-shadow: 0 12px 26px rgba(245,184,75,.18) !important;
    animation:none !important;
}
.invest-panel {
    background: var(--panel-2);
    border:1px solid rgba(245,184,75,.28);
    border-radius:8px;
    padding:24px;
    margin:18px 0 24px;
    position:relative;
    overflow:hidden;
    box-shadow:var(--shadow);
}
.invest-panel::before, .invest-panel::after { content:none; }
.invest-header { display:flex; align-items:center; gap:12px; margin-bottom:18px; position:relative; z-index:1; }
.invest-header-icon { font-size:1.35rem; width:42px; height:42px; border-radius:8px; background:var(--brand-soft); border:1px solid rgba(245,184,75,.28); display:flex; align-items:center; justify-content:center; }
.invest-header-title { font-size:1.15rem; font-weight:800; color:var(--text); }
.invest-header-sub { font-size:.82rem; color:var(--muted); margin-top:2px; }

.trade-summary { padding:18px; margin:16px 0; }
.trade-summary-title { font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; color:var(--brand); font-weight:900; margin-bottom:12px; }
.trade-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:10px; }
.trade-item { background:rgba(8,10,15,.5); border:1px solid var(--stroke); border-radius:8px; padding:11px 12px; }
.trade-label { font-size:.67rem; text-transform:uppercase; letter-spacing:.07em; color:var(--faint); margin-bottom:6px; font-weight:800; }
.trade-value { font-family:'JetBrains Mono',monospace; font-size:.95rem; font-weight:800; color:var(--text); overflow-wrap:anywhere; }
.trade-value.pos { color:var(--green); } .trade-value.neg { color:var(--red); } .trade-value.neutral { color:var(--brand); }

/* ---- SUPPORTING COMPONENTS ---- */
.risk-row { display:flex; gap:10px; align-items:center; margin-bottom:8px; }
.risk-badge { display:inline-flex; align-items:center; gap:6px; padding:5px 12px; border-radius:999px; font-size:.76rem; font-weight:900; letter-spacing:.04em; }
.risk-low { background:rgba(34,197,94,.12); color:#5ee68a; border:1px solid rgba(34,197,94,.28); }
.risk-med { background:rgba(245,184,75,.12); color:#f8d58a; border:1px solid rgba(245,184,75,.28); }
.risk-high { background:rgba(239,68,68,.12); color:#ff7b7b; border:1px solid rgba(239,68,68,.28); }
.insight-pill, .liq-warning { border-radius:8px; padding:12px 14px; margin-bottom:9px; font-size:.84rem; color:var(--muted); display:flex; align-items:flex-start; gap:9px; }
.insight-pill { background:rgba(90,167,255,.08); border:1px solid rgba(90,167,255,.2); }
.insight-pill .ic { display:none; }
.liq-warning { background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.28); color:#ffaaaa; }
.alloc-row { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
.alloc-coin { width:60px; font-size:.84rem; font-weight:800; color:var(--text); }
.alloc-bar-bg { flex:1; height:9px; background:rgba(148,163,184,.16); border-radius:999px; overflow:hidden; }
.alloc-bar-fill { height:100%; border-radius:999px; }
.alloc-pct { width:50px; text-align:right; font-family:'JetBrains Mono',monospace; font-size:.82rem; color:var(--text); font-weight:800; }
.footer { text-align:center; color:var(--faint); font-size:.74rem; padding:20px 0 6px; border-top:1px solid var(--stroke); margin-top:32px; }
.footer span { color:var(--muted); }

@media (max-width: 900px) {
    .block-container { padding: 1rem 1rem 2rem !important; }
    .hero-wrap { padding: 24px 22px; }
    .hero-title { font-size: 1.65rem; }
    .model-compare, .dash-grid { grid-template-columns:1fr; }
    .signal-banner, .ai-rec-box { align-items:flex-start; flex-direction:column; }
    .ai-rec-amount { margin-left:0; text-align:left; }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
#  SESSION STATE - active view
# =====================================================
if "view" not in st.session_state:
    st.session_state.view = "Predictor"
if "show_invest_panel" not in st.session_state:
    st.session_state.show_invest_panel = False

# =====================================================
#  SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("### CryptoSage AI")
    st.markdown("---")

    coin = st.selectbox(
        "Cryptocurrency",
        ["bitcoin","ethereum","litecoin","solana","cardano","dogecoin"],
        format_func=lambda x: {
            "bitcoin":  "Bitcoin (BTC)",
            "ethereum": "Ethereum (ETH)",
            "litecoin": "Litecoin (LTC)",
            "solana":   "Solana (SOL)",
            "cardano":  "Cardano (ADA)",
            "dogecoin": "Dogecoin (DOGE)"
        }[x]
    )
    st.markdown("---")
    display_currency = st.selectbox("Display Currency", ["INR", "USD"], index=0)
    st.caption("Charts switch currency; key prediction values show INR and USD together.")
    st.markdown("---")
    investment = st.number_input(f"Investment Amount ({'₹' if display_currency == 'INR' else '$'})", min_value=1000, max_value=10000000, value=10000, step=1000)
    days       = st.slider("Historical Data (Days)", 30, 365, 180)
    st.markdown("---")
    model_choice = st.radio("AI Prediction Model", ["LSTM","Linear Regression","Hybrid (LSTM + LR)"], index=2)
    st.markdown("---")
    show_bollinger   = st.toggle("Bollinger Bands",     value=True)
    show_rsi         = st.toggle("RSI Indicator",       value=True)
    show_macd        = st.toggle("MACD Chart",          value=True)
    show_volume      = st.toggle("Volume Distribution", value=True)
    show_model_cmp   = st.toggle("Model Comparison",    value=True)
    st.markdown("---")
    st.markdown("<div style='font-size:.72rem;color:#627084;text-align:center;line-height:1.55'>CryptoSage AI v3.0<br>Multi-currency crypto intelligence</div>", unsafe_allow_html=True)

# =====================================================
#  NAV BAR  (3 view buttons)
# =====================================================
VIEWS = ["Dashboard", "Predictor", "Manual Model"]
cols_nav = st.columns(len(VIEWS))
for col, v in zip(cols_nav, VIEWS):
    label = v
    active = "active" if st.session_state.view == label else ""
    if col.button(v, key=f"nav_{label}", use_container_width=True):
        st.session_state.view = label
        st.rerun()

VIEW = st.session_state.view
DISPLAY_CURRENCY = display_currency

# =====================================================
#  HELPERS - TECHNICAL INDICATORS
# =====================================================
COIN_SYMBOLS = {"bitcoin":"BTC","ethereum":"ETH","litecoin":"LTC","solana":"SOL","cardano":"ADA","dogecoin":"DOGE"}
symbol = COIN_SYMBOLS[coin]

@st.cache_data(ttl=300)
def get_usd_inr_rate():
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "tether", "vs_currencies": "inr"},
            timeout=10
        )
        rate = float(r.json()["tether"]["inr"])
        return rate if rate > 0 else 83.5
    except Exception:
        return 83.5

USD_INR_RATE = get_usd_inr_rate()

def convert_to_inr(value):
    return value * USD_INR_RATE

def convert_to_usd(value):
    return value / USD_INR_RATE

def format_currency(value, currency="INR"):
    if currency == "USD":
        return f"${value:,.2f}"
    return f"₹{value:,.2f}"

def both_currency_text(value, value_currency="INR"):
    inr_value = value if value_currency == "INR" else convert_to_inr(value)
    usd_value = value if value_currency == "USD" else convert_to_usd(value)
    return f"{format_currency(inr_value, 'INR')} / {format_currency(usd_value, 'USD')}"

def currency_value(inr_value, currency):
    return convert_to_usd(inr_value) if currency == "USD" else inr_value

def currency_tick_prefix(currency):
    return "$" if currency == "USD" else "₹"

def add_dual_currency_columns(df, source_currency):
    df = df.copy()
    if source_currency == "USD":
        df["price_usd"] = df["price"]
        df["price_inr"] = df["price"].apply(convert_to_inr)
    else:
        df["price_inr"] = df["price"]
        df["price_usd"] = df["price"].apply(convert_to_usd)
    return df

def set_active_currency(df, currency):
    df = df.copy()
    if currency == "USD" and "price_usd" in df.columns:
        df["price"] = df["price_usd"]
    elif "price_inr" in df.columns:
        df["price"] = df["price_inr"]
    return df

def format_compact_currency(value, currency="INR"):
    sign = "$" if currency == "USD" else "₹"
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"{sign}{value/1e12:.2f}T"
    if abs_value >= 1e9:
        return f"{sign}{value/1e9:.2f}B"
    if abs_value >= 1e6:
        return f"{sign}{value/1e6:.2f}M"
    return format_currency(value, currency)

def compute_rsi(s, period=14):
    d = s.diff(); g = d.clip(lower=0).rolling(period).mean(); l = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - (100/(1+g/l.replace(0,np.nan)))

def compute_macd(s, fast=12, slow=26, sig=9):
    ef = s.ewm(span=fast,adjust=False).mean(); es = s.ewm(span=slow,adjust=False).mean()
    m = ef-es; sl = m.ewm(span=sig,adjust=False).mean(); return m, sl, m-sl

def compute_bollinger(s, w=20, n=2):
    sma=s.rolling(w).mean(); std=s.rolling(w).std(); return sma+n*std, sma, sma-n*std

def sec(icon, text):
    st.markdown(f'<div class="sec-head"><div class="sec-text">{text}</div></div>', unsafe_allow_html=True)

def kpi_html(label, value, change=None, acc="linear-gradient(90deg,#6366f1,#818cf8)"):
    ch = ""
    if change is not None:
        cls = "up" if change>0 else ("dn" if change<0 else "ne")
        sgn = "+" if change>0 else ("-" if change<0 else "")
        ch  = f'<div class="kpi-ch {cls}">{sgn} {abs(change):.2f}%</div>'
    return f'<div class="kpi-card" style="--acc:{acc}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{ch}</div>'

# =====================================================
#  FETCH LIVE DATA  (cached 5 min)
# =====================================================
@st.cache_data(ttl=300)
def fetch_crypto_data(coin, days, display_currency):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    r   = requests.get(url, params={"vs_currency":"inr","days":days})
    d   = r.json()
    df  = pd.DataFrame(d["prices"], columns=["timestamp","price_inr"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    if "total_volumes" in d:
        vdf = pd.DataFrame(d["total_volumes"], columns=["timestamp","volume_inr"])
        vdf["timestamp"] = pd.to_datetime(vdf["timestamp"], unit="ms")
        vdf.set_index("timestamp", inplace=True)
        df = df.join(vdf, how="left")
    if "market_caps" in d:
        mdf = pd.DataFrame(d["market_caps"], columns=["timestamp","market_cap_inr"])
        mdf["timestamp"] = pd.to_datetime(mdf["timestamp"], unit="ms")
        mdf.set_index("timestamp", inplace=True)
        df = df.join(mdf, how="left")
    df["price_usd"] = df["price_inr"].apply(convert_to_usd)
    df["price"] = df["price_usd"] if display_currency == "USD" else df["price_inr"]
    if "volume_inr" in df.columns:
        df["volume_usd"] = df["volume_inr"].apply(convert_to_usd)
        df["volume"] = df["volume_usd"] if display_currency == "USD" else df["volume_inr"]
    if "market_cap_inr" in df.columns:
        df["market_cap_usd"] = df["market_cap_inr"].apply(convert_to_usd)
        df["market_cap"] = df["market_cap_usd"] if display_currency == "USD" else df["market_cap_inr"]
    return df

@st.cache_data(ttl=300)
def fetch_all_coins(display_currency):
    coins = ["bitcoin","ethereum","litecoin","solana","cardano","dogecoin"]
    rows  = []
    for c in coins:
        try:
            r = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{c}/market_chart",
                params={"vs_currency":"inr","days":2}
            )
            prices = r.json()["prices"]
            p_now_inr  = prices[-1][1];  p_24h = prices[max(0,len(prices)-25)][1]
            p_now = currency_value(p_now_inr, display_currency)
            chg    = ((p_now_inr-p_24h)/p_24h)*100
            rows.append({"coin":c,"symbol":COIN_SYMBOLS[c],"price":p_now,"price_inr":p_now_inr,"price_usd":convert_to_usd(p_now_inr),"change_24h":chg})
        except:
            pass
    return rows

# ===================================================
# ============  VIEW: DASHBOARD  ====================
# ===================================================
if VIEW == "Dashboard":

    st.markdown('<div class="hero-wrap"><div class="hero-badge">Live Market Overview</div><p class="hero-title"><span>CryptoSage</span> Dashboard</p><p class="hero-sub">Real-time snapshot of all tracked cryptocurrencies</p></div>', unsafe_allow_html=True)

    with st.spinner("Loading market snapshot..."):
        all_data = fetch_all_coins(DISPLAY_CURRENCY)

    if all_data:
        sec("ðŸª™", "Live Prices - All Coins")
        cols3 = st.columns(3)
        for i, row in enumerate(all_data):
            cls   = "up" if row["change_24h"]>=0 else "dn"
            arrow = "â–²" if row["change_24h"]>=0 else "â–¼"
            with cols3[i % 3]:
                st.markdown(f"""
                <div class="dash-coin-card">
                    <div class="dash-coin-top">
                        <div>
                            <div class="dash-coin-name">{row['coin'].capitalize()}</div>
                            <div class="dash-coin-sym">{row['symbol']}/{DISPLAY_CURRENCY}</div>
                        </div>
                        <span class="dash-coin-change {cls}">{arrow} {abs(row['change_24h']):.2f}%</span>
                    </div>
                    <div class="dash-coin-price">{format_currency(row['price'], DISPLAY_CURRENCY)}</div>
                </div>""", unsafe_allow_html=True)

    # Comparison bar chart
    sec("ðŸ“Š", "24h Price Change Comparison")
    if all_data:
        bar_fig = go.Figure(go.Bar(
            x=[r["symbol"] for r in all_data],
            y=[r["change_24h"] for r in all_data],
            marker_color=["#34d399" if r["change_24h"]>=0 else "#f87171" for r in all_data],
            text=[f"{r['change_24h']:+.2f}%" for r in all_data],
            textposition="outside"
        ))
        bar_fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,10,20,.6)", height=300,
            font=dict(family="Space Grotesk",color="#64748b",size=12),
            yaxis=dict(title="24h Change %", gridcolor="#0f172a", zeroline=True, zerolinecolor="#1e293b"),
            xaxis=dict(gridcolor="#0f172a"),
            margin=dict(l=10,r=10,t=20,b=10), showlegend=False
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # Selected coin mini chart
    sec("ðŸ“ˆ", f"{symbol} - {days}d Price History")
    with st.spinner("Loading chart..."):
        df_dash = fetch_crypto_data(coin, days, DISPLAY_CURRENCY)
    dash_fig = go.Figure()
    dash_fig.add_trace(go.Scatter(
        x=df_dash.index, y=df_dash["price"],
        mode="lines", line=dict(color="#f59e0b", width=2.2), fill="tozeroy",
        fillcolor="rgba(245,158,11,.05)", name="Price",
        hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{currency_tick_prefix(DISPLAY_CURRENCY)}%{{y:,.2f}}<extra></extra>"
    ))
    dash_fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,.6)", height=320,
        font=dict(family="Space Grotesk",color="#64748b",size=11),
        xaxis=dict(gridcolor="#0f172a"), yaxis=dict(gridcolor="#0f172a",tickprefix=currency_tick_prefix(DISPLAY_CURRENCY),tickformat=",.0f"),
        margin=dict(l=10,r=10,t=10,b=10), showlegend=False
    )
    st.plotly_chart(dash_fig, use_container_width=True)

    # KPI row for selected coin
    cp    = df_dash["price"].iloc[-1]
    p1d   = df_dash["price"].iloc[-min(len(df_dash),24)]
    ch1d  = ((cp-p1d)/p1d)*100
    hi    = df_dash["price"].max()
    lo    = df_dash["price"].min()
    vol_d = df_dash["price"].pct_change().std()
    rsi_d = compute_rsi(df_dash["price"]).iloc[-1]
    sec("ðŸ“‹", f"{symbol} Key Stats")
    cards = (
        kpi_html("Current Price",     both_currency_text(cp, DISPLAY_CURRENCY),    ch1d,  "linear-gradient(90deg,#6366f1,#818cf8)") +
        kpi_html(f"{days}d High",     format_currency(hi, DISPLAY_CURRENCY),    None,  "linear-gradient(90deg,#34d399,#6ee7b7)") +
        kpi_html(f"{days}d Low",      format_currency(lo, DISPLAY_CURRENCY),    None,  "linear-gradient(90deg,#f87171,#fca5a5)") +
        kpi_html("RSI (14)",          f"{rsi_d:.1f}",   None,  "linear-gradient(90deg,#a78bfa,#c4b5fd)") +
        kpi_html("Daily Volatility",  f"{vol_d*100:.2f}%", None,"linear-gradient(90deg,#f59e0b,#fbbf24)")
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">Built by <span>Nagesh</span> | CryptoSage AI v3.0 | <span>Not financial advice</span></div>', unsafe_allow_html=True)


# ===================================================
# ============  VIEW: PREDICTOR  ====================
# ===================================================
elif VIEW == "Predictor":

    with st.spinner("Fetching live market data..."):
        df = fetch_crypto_data(coin, days, DISPLAY_CURRENCY)

    current_price = df["price"].iloc[-1]
    p1d_ago       = df["price"].iloc[-min(len(df),24)]
    change_1d     = ((current_price-p1d_ago)/p1d_ago)*100
    p7d_ago       = df["price"].iloc[-min(len(df),200)]
    change_7d     = ((current_price-p7d_ago)/p7d_ago)*100
    high_p        = df["price"].max()
    low_p         = df["price"].min()

    df["rsi"]                            = compute_rsi(df["price"])
    df["ma20"]                           = df["price"].rolling(20).mean()
    df["ma50"]                           = df["price"].rolling(50).mean()
    df["bb_upper"],df["bb_mid"],df["bb_lower"] = compute_bollinger(df["price"])
    df["macd"],df["macd_signal"],df["macd_hist"] = compute_macd(df["price"])

    rsi_current = df["rsi"].iloc[-1]
    rsi_label   = "Overbought" if rsi_current>70 else ("Oversold" if rsi_current<30 else "Neutral")

    mc_str = "-"
    if "market_cap" in df.columns:
        mc = df["market_cap"].iloc[-1]
        mc_str = format_compact_currency(mc, DISPLAY_CURRENCY)

    chg1d_col = "#34d399" if change_1d>=0 else "#f87171"
    chg1d_arr = "â–²" if change_1d>=0 else "â–¼"

    # HERO
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-badge">AI Price Predictor | Live</div>
        <p class="hero-title">{symbol} / {DISPLAY_CURRENCY} &nbsp;<span>CryptoSage</span></p>
        <p class="hero-sub">Real-time forecasting | Technical analysis | Risk assessment</p>
        <div class="hero-stats">
            <div class="hero-stat"><span class="hero-stat-val">{format_currency(current_price, DISPLAY_CURRENCY)}</span><span class="hero-stat-lbl">Current Price</span></div>
            <div class="hero-divider"></div>
            <div class="hero-stat"><span class="hero-stat-val" style="color:{chg1d_col}">{chg1d_arr} {abs(change_1d):.2f}%</span><span class="hero-stat-lbl">24h Change</span></div>
            <div class="hero-divider"></div>
            <div class="hero-stat"><span class="hero-stat-val">{rsi_current:.1f}</span><span class="hero-stat-lbl">RSI | {rsi_label}</span></div>
            <div class="hero-divider"></div>
            <div class="hero-stat"><span class="hero-stat-val">{mc_str}</span><span class="hero-stat-lbl">Market Cap</span></div>
        </div>
    </div>""", unsafe_allow_html=True)

    # KPI CARDS
    cards_html = (
        kpi_html("Current Price",  f"{format_currency(current_price, DISPLAY_CURRENCY)}", change_1d, "linear-gradient(90deg,#6366f1,#818cf8)") +
        kpi_html(f"{days}d High",  format_currency(high_p, DISPLAY_CURRENCY),        None,      "linear-gradient(90deg,#34d399,#6ee7b7)") +
        kpi_html(f"{days}d Low",   format_currency(low_p, DISPLAY_CURRENCY),         None,      "linear-gradient(90deg,#f87171,#fca5a5)") +
        kpi_html("7d Change",      f"{change_7d:+.2f}%",     change_7d, "linear-gradient(90deg,#f59e0b,#fbbf24)") +
        kpi_html("RSI (14)",       f"{rsi_current:.1f}",     None,      "linear-gradient(90deg,#a78bfa,#c4b5fd)")
    )
    st.markdown(f'<div class="kpi-grid">{cards_html}</div>', unsafe_allow_html=True)

    # PRICE CHART
    sec("ðŸ“ˆ", "Price Chart")
    pfig = go.Figure()
    if show_bollinger:
        pfig.add_trace(go.Scatter(x=df.index,y=df["bb_upper"],mode="lines",line=dict(color="rgba(99,102,241,.3)",width=1,dash="dot"),name="BB Upper"))
        pfig.add_trace(go.Scatter(x=df.index,y=df["bb_lower"],mode="lines",line=dict(color="rgba(99,102,241,.3)",width=1,dash="dot"),fill="tonexty",fillcolor="rgba(99,102,241,.04)",name="BB Band"))
    pfig.add_trace(go.Scatter(x=df.index,y=df["price"],mode="lines",line=dict(color="#f59e0b",width=2.5),name=f"{symbol} Price",hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{currency_tick_prefix(DISPLAY_CURRENCY)}%{{y:,.2f}}<extra></extra>"))
    pfig.add_trace(go.Scatter(x=df.index,y=df["ma20"],mode="lines",line=dict(color="#818cf8",width=1.2,dash="dash"),name="MA 20"))
    pfig.add_trace(go.Scatter(x=df.index,y=df["ma50"],mode="lines",line=dict(color="#34d399",width=1.2,dash="dash"),name="MA 50"))
    pfig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",height=420,
                       font=dict(family="Space Grotesk",color="#64748b",size=12),
                       legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0,bgcolor="rgba(0,0,0,0)"),
                       xaxis=dict(gridcolor="#0f172a"),yaxis=dict(gridcolor="#0f172a",tickprefix=currency_tick_prefix(DISPLAY_CURRENCY),tickformat=",.0f"),
                       hovermode="x unified",margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(pfig, use_container_width=True)

    # RSI + MACD
    if show_rsi or show_macd:
        n = int(show_rsi)+int(show_macd)
        titles = (["RSI (14)"] if show_rsi else []) + (["MACD"] if show_macd else [])
        ifig = make_subplots(rows=n,cols=1,shared_xaxes=True,row_heights=[1]*n,subplot_titles=titles,vertical_spacing=.08)
        r = 1
        if show_rsi:
            ifig.add_trace(go.Scatter(x=df.index,y=df["rsi"],line=dict(color="#a78bfa",width=1.8),name="RSI"),row=r,col=1)
            ifig.add_hline(y=70,line=dict(color="#f87171",dash="dot",width=1),row=r,col=1)
            ifig.add_hline(y=30,line=dict(color="#34d399",dash="dot",width=1),row=r,col=1)
            ifig.add_hrect(y0=70,y1=100,fillcolor="rgba(248,113,113,.04)",line_width=0,row=r,col=1)
            ifig.add_hrect(y0=0, y1=30, fillcolor="rgba(52,211,153,.04)", line_width=0,row=r,col=1)
            r+=1
        if show_macd:
            hc = ["#34d399" if v>=0 else "#f87171" for v in df["macd_hist"].fillna(0)]
            ifig.add_trace(go.Bar(x=df.index,y=df["macd_hist"],marker_color=hc,name="Histogram",opacity=.6),row=r,col=1)
            ifig.add_trace(go.Scatter(x=df.index,y=df["macd"],line=dict(color="#6366f1",width=1.5),name="MACD"),row=r,col=1)
            ifig.add_trace(go.Scatter(x=df.index,y=df["macd_signal"],line=dict(color="#f59e0b",width=1.5,dash="dash"),name="Signal"),row=r,col=1)
        ifig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",
                           height=200*n,font=dict(family="Space Grotesk",color="#64748b",size=11),
                           legend=dict(orientation="h",yanchor="bottom",y=1.01,x=0,bgcolor="rgba(0,0,0,0)"),
                           xaxis=dict(gridcolor="#0f172a"),yaxis=dict(gridcolor="#0f172a"),
                           margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(ifig, use_container_width=True)

    # VOLUME
    if show_volume and "volume" in df.columns and not df["volume"].isna().all():
        sec("ðŸ“¦","Volume Distribution")
        vfig = go.Figure()
        vfig.add_trace(go.Bar(
            x=df.index, y=df["volume"],
            marker_color=df["price"].pct_change().fillna(0).apply(lambda x:"rgba(52,211,153,.5)" if x>=0 else "rgba(248,113,113,.5)"),
            hovertemplate="<b>%{x|%d %b}</b><br>Vol: %{y:,.0f}<extra></extra>"
        ))
        vfig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",
                           height=200,showlegend=False,margin=dict(l=10,r=10,t=10,b=10),
                           xaxis=dict(gridcolor="#0f172a"),yaxis=dict(gridcolor="#0f172a",tickformat=".2s"))
        st.plotly_chart(vfig, use_container_width=True)

    # ML PREPROCESSING
    prices_arr = df["price"].values.reshape(-1,1)
    scaler     = MinMaxScaler()
    scaled     = scaler.fit_transform(prices_arr)
    window_size= 15
    X,y_arr    = [],[]
    for i in range(window_size,len(scaled)):
        X.append(scaled[i-window_size:i]); y_arr.append(scaled[i])
    X = np.array(X); y_arr = np.array(y_arr)
    split      = int(len(X)*.8)
    X_tr,X_te  = X[:split],X[split:]
    y_tr,y_te  = y_arr[:split],y_arr[split:]

    lr_model = LinearRegression()
    lr_model.fit(X_tr.reshape(X_tr.shape[0],-1), y_tr)
    lr_p_sc  = lr_model.predict(X_te.reshape(X_te.shape[0],-1)).flatten()
    lr_preds = scaler.inverse_transform(lr_p_sc.reshape(-1,1)).flatten()

    lstm_model = None
    if model_choice in ["LSTM","Hybrid (LSTM + LR)"]:
        with st.spinner("Training LSTM..."):
            lstm_model = Sequential([
                LSTM(64,return_sequences=True,input_shape=(X.shape[1],1)),
                Dropout(.2), LSTM(32), Dropout(.2), Dense(16), Dense(1)
            ])
            lstm_model.compile(optimizer="adam",loss="huber")
            lstm_model.fit(X_tr,y_tr,epochs=8,batch_size=32,verbose=0)
        lstm_p_sc= lstm_model.predict(X_te,verbose=0).flatten()
        lstm_preds= scaler.inverse_transform(lstm_p_sc.reshape(-1,1)).flatten()

    last_w  = scaled[-window_size:]
    X_pred  = np.array([last_w])
    lr_next_sc = lr_model.predict(X_pred.reshape(1,-1))[0][0]
    lr_next = scaler.inverse_transform([[lr_next_sc]])[0][0]

    if lstm_model:
        lstm_next_sc= lstm_model.predict(X_pred,verbose=0)[0][0]
        lstm_next   = scaler.inverse_transform([[lstm_next_sc]])[0][0]
    else:
        lstm_next = lr_next

    if   model_choice=="Linear Regression":  predicted_price = lr_next
    elif model_choice=="LSTM":               predicted_price = lstm_next
    else:                                    predicted_price = (lr_next+lstm_next)/2

    change_percent = ((predicted_price-current_price)/current_price)*100
    y_true         = scaler.inverse_transform(y_te.reshape(-1,1)).flatten()
    lr_mae  = mean_absolute_error(y_true,lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_true,lr_preds))
    lr_mape = np.mean(np.abs((y_true-lr_preds)/y_true))*100
    lr_acc  = max(0,100-lr_mape)
    lstm_acc= None
    if lstm_model:
        lstm_mape = np.mean(np.abs((y_true-lstm_preds)/y_true))*100
        lstm_acc  = max(0,100-lstm_mape)

    # MODEL COMPARISON
    if show_model_cmp:
        sec("ðŸ¤–","Model Comparison")
        hybrid_next = (lr_next+lstm_next)/2
        is_lr = model_choice=="Linear Regression"
        is_ls = model_choice=="LSTM"
        is_hy = model_choice=="Hybrid (LSTM + LR)"
        st.markdown(f"""
        <div class="model-compare">
            <div class="model-card {'active' if is_lr else ''}">
                <div class="model-name">Linear Regression</div>
                <div class="model-pred">{format_currency(lr_next, DISPLAY_CURRENCY)}</div>
                <div class="model-tag {'at' if is_lr else 'it'}">{'Active' if is_lr else f'Acc {lr_acc:.1f}%'}</div>
            </div>
            <div class="model-card {'active' if is_ls else ''}">
                <div class="model-name">LSTM Neural Net</div>
                <div class="model-pred">{format_currency(lstm_next, DISPLAY_CURRENCY)}</div>
                <div class="model-tag {'at' if is_ls else 'it'}">{'Active' if is_ls else (f'Acc {lstm_acc:.1f}%' if lstm_acc else '-')}</div>
            </div>
            <div class="model-card {'active' if is_hy else ''}">
                <div class="model-name">Hybrid Ensemble</div>
                <div class="model-pred">{format_currency(hybrid_next, DISPLAY_CURRENCY)}</div>
                <div class="model-tag {'at' if is_hy else 'it'}">{'Active' if is_hy else 'Avg of both'}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        sec("ðŸ”¬","Backtest - Actual vs Predicted")
        bt_dates = df.index[window_size+split:]
        bfig = go.Figure()
        bfig.add_trace(go.Scatter(x=bt_dates,y=y_true,line=dict(color="#f59e0b",width=2),name="Actual"))
        bfig.add_trace(go.Scatter(x=bt_dates,y=lr_preds,line=dict(color="#6366f1",width=1.5,dash="dash"),name="LR"))
        if lstm_model:
            bfig.add_trace(go.Scatter(x=bt_dates,y=lstm_preds,line=dict(color="#34d399",width=1.5,dash="dot"),name="LSTM"))
        bfig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",
                           height=300,font=dict(family="Space Grotesk",color="#64748b",size=11),
                           legend=dict(orientation="h",yanchor="bottom",y=1.01,x=0,bgcolor="rgba(0,0,0,0)"),
                           xaxis=dict(gridcolor="#0f172a"),yaxis=dict(gridcolor="#0f172a",tickprefix=currency_tick_prefix(DISPLAY_CURRENCY),tickformat=",.0f"),
                           margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(bfig, use_container_width=True)
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("LR MAE",      format_currency(lr_mae, DISPLAY_CURRENCY))
        mc2.metric("LR RMSE",     format_currency(lr_rmse, DISPLAY_CURRENCY))
        mc3.metric("LR Accuracy", f"{lr_acc:.1f}%")
        if lstm_acc: mc4.metric("LSTM Accuracy", f"{lstm_acc:.1f}%")

    # SIGNAL BANNER
    sec("ðŸ”®","Next Candle Prediction")
    if   predicted_price > current_price*1.02:
        sig,sc,si,sd = "BUY","buy","",  f"{symbol} shows upward momentum. Model projects a {change_percent:.2f}% gain."
    elif predicted_price < current_price*0.98:
        sig,sc,si,sd = "SELL","sell","", f"{symbol} shows downward pressure. Model projects a {abs(change_percent):.2f}% decline."
    else:
        sig,sc,si,sd = "HOLD","hold","", f"{symbol} is consolidating. Price within +/-2%. Wait for a clearer signal."

    st.markdown(f"""
    <div class="signal-banner {sc}">
        <div class="sig-icon">{si}</div>
        <div><div class="sig-title {sc}">{sig} Signal - {model_choice}</div><div class="sig-desc">{sd}</div></div>
        <div style="margin-left:auto;text-align:right">
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;color:#f1f5f9;font-weight:600">{format_currency(predicted_price, DISPLAY_CURRENCY)}</div>
            <div style="font-size:.73rem;color:#475569">Predicted Price</div>
        </div>
    </div>""", unsafe_allow_html=True)

    pc1,pc2,pc3,pc4 = st.columns(4)
    pc1.metric("Current Price",   both_currency_text(current_price, DISPLAY_CURRENCY))
    pc2.metric("Predicted Price", both_currency_text(predicted_price, DISPLAY_CURRENCY))
    pc3.metric("Expected Change", f"{change_percent:+.2f}%")
    pc4.metric("Active Model",    model_choice.split()[0])

    # PORTFOLIO
    sec("ðŸ’¼","Portfolio Simulation")
    coins_bought = investment/current_price
    future_value = coins_bought*predicted_price
    profit       = future_value-investment
    roi          = (profit/investment)*100
    pp1,pp2,pp3,pp4 = st.columns(4)
    pp1.metric("Investment",   both_currency_text(investment, DISPLAY_CURRENCY))
    pp2.metric("Coins Bought", f"{coins_bought:.6f} {symbol}")
    pp3.metric("Future Value", both_currency_text(future_value, DISPLAY_CURRENCY))
    pp4.metric("P&L",          both_currency_text(profit, DISPLAY_CURRENCY), f"{roi:+.2f}%")

    st.markdown("**Scenario Analysis**")
    sc_df = pd.DataFrame({
        "Scenario":       ["Bear (-10%)","Base (Model)","Bull (+10%)","Bull (+20%)"],
        "Price":          [current_price*.9,predicted_price,current_price*1.1,current_price*1.2],
        "Portfolio Value":[coins_bought*current_price*.9,future_value,coins_bought*current_price*1.1,coins_bought*current_price*1.2],
        "P&L":            [coins_bought*current_price*.9-investment,profit,coins_bought*current_price*1.1-investment,coins_bought*current_price*1.2-investment]
    })
    sc_df["Price"]            = sc_df["Price"].apply(lambda x:format_currency(x, DISPLAY_CURRENCY))
    sc_df["Portfolio Value"]  = sc_df["Portfolio Value"].apply(lambda x:format_currency(x, DISPLAY_CURRENCY))
    sc_df["P&L"]              = sc_df["P&L"].apply(lambda x:f"{currency_tick_prefix(DISPLAY_CURRENCY)}{x:+,.2f}")
    st.dataframe(sc_df, use_container_width=True, hide_index=True)

    # =====================================================
    #  ðŸ’¸ INVEST NOW - BUTTON + ADVANCED PANEL
    # =====================================================
    st.markdown('<div class="invest-btn-wrap">', unsafe_allow_html=True)
    if st.button("Invest Now", key="invest_now_btn", use_container_width=True):
        st.session_state.show_invest_panel = not st.session_state.show_invest_panel
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.show_invest_panel:

        st.markdown("""
        <div class="invest-panel">
            <div class="invest-header">
                <div class="invest-header-icon">AI</div>
                <div>
                    <div class="invest-header-title">Advanced Investment Panel</div>
                    <div class="invest-header-sub">Configure your trade - leverage, risk, DCA, SIP & scenario simulation</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # ---------------- INPUT CONTROLS ----------------
        ic1, ic2, ic3 = st.columns(3)
        inv_amount = ic1.number_input("Investment Amount", min_value=100, max_value=100000000, value=int(investment), step=500, key="inv_amount")
        leverage   = ic2.select_slider("Leverage", options=[1,2,5,10], value=1, format_func=lambda x:f"{x}x", key="inv_leverage")
        risk_level_sel = ic3.selectbox("Risk Level", ["Low","Medium","High"], index=1, key="inv_risk")

        ic4, ic5, ic6 = st.columns(3)
        take_profit_pct = ic4.number_input("Take Profit %", min_value=0.5, max_value=500.0, value=10.0, step=0.5, key="inv_tp")
        stop_loss_pct   = ic5.number_input("Stop Loss %",   min_value=0.5, max_value=99.0,  value=5.0,  step=0.5, key="inv_sl")
        duration_sel    = ic6.selectbox("Duration", ["1 Day","7 Days","30 Days","Custom"], index=1, key="inv_duration")

        custom_days = None
        if duration_sel == "Custom":
            custom_days = st.slider("Custom Duration (Days)", 1, 365, 14, key="inv_custom_days")

        ic7, ic8, ic9 = st.columns(3)
        auto_compound = ic7.toggle("Auto-Compound", value=False, key="inv_compound")
        dca_enabled   = ic8.toggle("Dollar Cost Averaging (DCA)", value=False, key="inv_dca")
        sip_enabled   = ic9.toggle("SIP Mode", value=False, key="inv_sip")

        sip_freq = None
        dca_chunks = None
        if sip_enabled:
            sip_freq = st.selectbox("SIP Frequency", ["Daily","Weekly","Monthly"], index=2, key="inv_sip_freq")
        if dca_enabled:
            dca_chunks = st.slider("DCA - Split Investment Into (chunks)", 2, 30, 5, key="inv_dca_chunks")

        target_price = st.number_input(
            f"Target Price - {symbol}", min_value=0.0,
            value=float(round(predicted_price,2)), step=10.0, key="inv_target_price"
        )

        # ---------------- CORE CALCULATIONS ----------------
        entry_price = current_price
        duration_map = {"1 Day":1, "7 Days":7, "30 Days":30, "Custom":custom_days or 1}
        duration_days = duration_map[duration_sel]

        # Position size with leverage
        position_size = inv_amount * leverage
        units_held    = position_size / entry_price

        # Predicted exit price (model-based, scaled by leverage exposure but P&L capped by margin)
        price_change_pct = change_percent  # from model
        exit_price_model = predicted_price

        # Raw P&L at predicted price (leveraged)
        raw_pnl   = units_held * (exit_price_model - entry_price)
        leveraged_roi = (raw_pnl / inv_amount) * 100 if inv_amount > 0 else 0

        # Auto-compound projection (simple daily compounding of ROI over duration)
        daily_growth_rate = price_change_pct / 100  # treat as overall period rate baseline
        if auto_compound and duration_days > 0:
            per_day_rate = (1 + leveraged_roi/100) ** (1/duration_days) - 1
            compounded_value = inv_amount * ((1 + per_day_rate) ** duration_days)
        else:
            compounded_value = inv_amount + raw_pnl

        estimated_profit = compounded_value - inv_amount
        roi_pct = (estimated_profit / inv_amount) * 100 if inv_amount > 0 else 0

        # Take profit / stop loss prices
        tp_price = entry_price * (1 + take_profit_pct/100)
        sl_price = entry_price * (1 - stop_loss_pct/100)

        # Risk-reward ratio
        reward = abs(tp_price - entry_price)
        risk   = abs(entry_price - sl_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0

        # Liquidation price (for leverage > 1)
        # Approx: liquidation when loss = margin (100% of invested amount / leverage exposure)
        if leverage > 1:
            liq_price = entry_price * (1 - 1/leverage)
        else:
            liq_price = None

        # Break-even point (accounting roughly for a flat 0.1% fee on entry+exit)
        fee_pct = 0.001
        breakeven_price = entry_price * (1 + fee_pct*2) / leverage * leverage  # entry+exit fee adjustment
        breakeven_price = entry_price * (1 + fee_pct*2)

        # ---------------- AI RECOMMENDATION ENGINE ----------------
        if   model_choice == "Linear Regression": model_acc = lr_acc
        elif model_choice == "LSTM":              model_acc = lstm_acc if lstm_acc else lr_acc
        else:                                      model_acc = ((lr_acc + (lstm_acc or lr_acc))/2)

        confidence = max(0, min(100, model_acc))

        if change_percent > 1.5 and confidence > 55:
            ai_action, ai_cls, ai_icon = "BUY", "buy", ""
            ai_desc = f"Model predicts +{change_percent:.2f}% move with {confidence:.1f}% confidence - favorable entry."
        elif change_percent < -1.5 and confidence > 55:
            ai_action, ai_cls, ai_icon = "SELL", "sell", ""
            ai_desc = f"Model predicts {change_percent:.2f}% move with {confidence:.1f}% confidence - consider exiting or shorting."
        else:
            ai_action, ai_cls, ai_icon = "HOLD", "hold", ""
            ai_desc = f"Model signal is weak ({change_percent:+.2f}%, {confidence:.1f}% confidence) - wait for clearer trend."

        # Safe investment amount suggestion based on risk level
        risk_caps = {"Low": 0.05, "Medium": 0.15, "High": 0.35}
        # crude "available capital" proxy = inv_amount as baseline capital reference
        suggested_safe_amount = inv_amount * risk_caps[risk_level_sel] / max(leverage,1) * 10
        suggested_safe_amount = min(suggested_safe_amount, inv_amount * (1.5 if risk_level_sel=="High" else 1.0))

        st.markdown(f"""
        <div class="ai-rec-box {ai_cls}">
            <div class="ai-rec-icon">{ai_icon}</div>
            <div>
                <div class="ai-rec-title {ai_cls}">AI Recommendation: {ai_action}</div>
                <div class="ai-rec-desc">{ai_desc}</div>
            </div>
            <div class="ai-rec-amount">
                <div class="ai-rec-amount-val">{format_currency(suggested_safe_amount, DISPLAY_CURRENCY)}</div>
                <div class="ai-rec-amount-lbl">Suggested Safe Amount ({risk_level_sel} Risk)</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ---------------- LIVE TRADE SUMMARY ----------------
        pnl_cls = "pos" if estimated_profit >= 0 else "neg"
        liq_html = f'{format_currency(liq_price, DISPLAY_CURRENCY)}' if liq_price is not None else "-  (1x, no liq.)"

        st.markdown(f"""
        <div class="trade-summary">
            <div class="trade-summary-title">Live Trade Summary - {symbol}/{DISPLAY_CURRENCY}</div>
            <div class="trade-grid">
                <div class="trade-item"><div class="trade-label">Entry Price</div><div class="trade-value">{format_currency(entry_price, DISPLAY_CURRENCY)}</div></div>
                <div class="trade-item"><div class="trade-label">Predicted Price</div><div class="trade-value neutral">{format_currency(exit_price_model, DISPLAY_CURRENCY)}</div></div>
                <div class="trade-item"><div class="trade-label">Target Price</div><div class="trade-value">{format_currency(target_price, DISPLAY_CURRENCY)}</div></div>
                <div class="trade-item"><div class="trade-label">Take Profit</div><div class="trade-value pos">{format_currency(tp_price, DISPLAY_CURRENCY)}</div></div>
                <div class="trade-item"><div class="trade-label">Stop Loss</div><div class="trade-value neg">{format_currency(sl_price, DISPLAY_CURRENCY)}</div></div>
                <div class="trade-item"><div class="trade-label">Break-even</div><div class="trade-value neutral">{format_currency(breakeven_price, DISPLAY_CURRENCY)}</div></div>
                <div class="trade-item"><div class="trade-label">Liquidation Price</div><div class="trade-value neg">{liq_html}</div></div>
                <div class="trade-item"><div class="trade-label">Position Size</div><div class="trade-value">{format_currency(position_size, DISPLAY_CURRENCY)}</div></div>
                <div class="trade-item"><div class="trade-label">Units Held</div><div class="trade-value">{units_held:.6f} {symbol}</div></div>
                <div class="trade-item"><div class="trade-label">Est. Profit/Loss</div><div class="trade-value {pnl_cls}">{format_currency(estimated_profit, DISPLAY_CURRENCY)}</div></div>
                <div class="trade-item"><div class="trade-label">ROI</div><div class="trade-value {pnl_cls}">{roi_pct:+.2f}%</div></div>
                <div class="trade-item"><div class="trade-label">Risk-Reward Ratio</div><div class="trade-value">1 : {risk_reward_ratio:.2f}</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

        if liq_price is not None:
            dist_to_liq = ((entry_price - liq_price)/entry_price)*100
            st.markdown(f"""
            <div class="liq-warning"><div><b>Warning: Leverage {leverage}x active.</b> Price needs to drop only <b>{dist_to_liq:.2f}%</b> to {format_currency(liq_price, DISPLAY_CURRENCY)} to trigger liquidation. Manage your stop loss carefully.</div></div>
            """, unsafe_allow_html=True)

        # DCA / SIP info
        if dca_enabled or sip_enabled:
            note_parts = []
            if dca_enabled:
                chunk_amt = inv_amount / dca_chunks
                note_parts.append(f"DCA: {format_currency(chunk_amt, DISPLAY_CURRENCY)} split across {dca_chunks} buys")
            if sip_enabled:
                note_parts.append(f"SIP: {format_currency(inv_amount, DISPLAY_CURRENCY)} recurring {sip_freq.lower()}")
            st.markdown(f'<div class="insight-pill"><span class="ic">ðŸ“†</span><span>{" | ".join(note_parts)} - reduces timing risk via systematic entries.</span></div>', unsafe_allow_html=True)

        if auto_compound:
            st.markdown(f'<div class="insight-pill"><span class="ic">ðŸ”</span><span>Auto-compound enabled - projected value compounds daily over {duration_days} day(s) toward {format_currency(compounded_value, DISPLAY_CURRENCY)}.</span></div>', unsafe_allow_html=True)

        # ---------------- SCENARIO SIMULATION ----------------
        sec("ðŸŽ²","Scenario Simulation")
        scenario_moves = {"Bear Market": -0.20, "Neutral Market": 0.0, "Bull Market": 0.25, "Extreme Bull Case": 0.60}
        scen_rows = []
        for sname, move in scenario_moves.items():
            sc_price = entry_price * (1 + move)
            sc_pnl   = units_held * (sc_price - entry_price)
            sc_roi   = (sc_pnl/inv_amount)*100 if inv_amount>0 else 0
            scen_rows.append({"Scenario":sname, "Price Move":f"{move*100:+.0f}%", "Price":f"{format_currency(sc_price, DISPLAY_CURRENCY)}",
                               "P&L":f"{currency_tick_prefix(DISPLAY_CURRENCY)}{sc_pnl:+,.2f}", "ROI":f"{sc_roi:+.2f}%"})
        st.dataframe(pd.DataFrame(scen_rows), use_container_width=True, hide_index=True)

        # ---------------- VISUAL CHARTS ----------------
        sec("ðŸ“Š","Profit Growth, Risk & ROI Projection")

        # Profit growth over duration (linear interpolation from 0 to estimated_profit)
        days_axis = np.arange(0, duration_days+1)
        if auto_compound and duration_days > 0:
            growth_curve = [inv_amount*((1+per_day_rate)**d) - inv_amount for d in days_axis]
        else:
            growth_curve = [estimated_profit * (d/duration_days if duration_days>0 else 1) for d in days_axis]

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            gfig = go.Figure()
            gfig.add_trace(go.Scatter(x=days_axis, y=growth_curve, mode="lines+markers",
                                       line=dict(color="#34d399" if estimated_profit>=0 else "#f87171", width=2.5),
                                       fill="tozeroy", fillcolor="rgba(52,211,153,.08)", name="Profit"))
            gfig.add_hline(y=0, line=dict(color="#475569", width=1, dash="dot"))
            gfig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,20,.6)",
                               height=260, title=dict(text="Profit Growth Projection", font=dict(size=12,color="#64748b")),
                               xaxis=dict(title="Day", gridcolor="#0f172a"), yaxis=dict(title="Profit", gridcolor="#0f172a"),
                               showlegend=False, margin=dict(l=10,r=10,t=36,b=10))
            st.plotly_chart(gfig, use_container_width=True)

        with col_chart2:
            # Risk exposure donut: margin used vs leverage exposure
            exposure_labels = ["Your Capital","Leveraged Exposure"] if leverage>1 else ["Your Capital"]
            exposure_values = [inv_amount, position_size-inv_amount] if leverage>1 else [inv_amount]
            rfig2 = go.Figure(go.Pie(labels=exposure_labels, values=exposure_values, hole=.62,
                                      marker=dict(colors=["#6366f1","#f87171"]), textinfo="label+percent"))
            rfig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 height=260, title=dict(text="Risk Exposure", font=dict(size=12,color="#64748b")),
                                 showlegend=False, margin=dict(l=10,r=10,t=36,b=10),
                                 annotations=[dict(text=f"{leverage}x", x=0.5, y=0.5, font=dict(size=18,color="#f1f5f9"), showarrow=False)])
            st.plotly_chart(rfig2, use_container_width=True)

        # ROI projection across price targets
        roi_price_range = np.linspace(entry_price*0.7, entry_price*1.5, 40)
        roi_curve = [((units_held*(p-entry_price))/inv_amount)*100 for p in roi_price_range]
        roifig = go.Figure()
        roifig.add_trace(go.Scatter(x=roi_price_range, y=roi_curve, mode="lines",
                                     line=dict(color="#818cf8", width=2.5), fill="tozeroy", fillcolor="rgba(129,140,248,.07)", name="ROI"))
        roifig.add_vline(x=entry_price, line=dict(color="#f59e0b", width=1.5, dash="dash"), annotation_text="Entry")
        roifig.add_vline(x=tp_price, line=dict(color="#34d399", width=1.5, dash="dot"), annotation_text="TP")
        roifig.add_vline(x=sl_price, line=dict(color="#f87171", width=1.5, dash="dot"), annotation_text="SL")
        roifig.add_vline(x=target_price, line=dict(color="#a78bfa", width=1.5, dash="dash"), annotation_text="Target")
        if liq_price is not None:
            roifig.add_vline(x=liq_price, line=dict(color="#ef4444", width=1.5, dash="dot"), annotation_text="Liq")
        roifig.add_hline(y=0, line=dict(color="#475569", width=1, dash="dot"))
        roifig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,10,20,.6)",
                              height=300, title=dict(text="ROI % vs Price Movement", font=dict(size=12,color="#64748b")),
                              xaxis=dict(title="Price", gridcolor="#0f172a", tickprefix=currency_tick_prefix(DISPLAY_CURRENCY), tickformat=",.0f"),
                              yaxis=dict(title="ROI %", gridcolor="#0f172a"),
                              showlegend=False, margin=dict(l=10,r=10,t=36,b=10))
        st.plotly_chart(roifig, use_container_width=True)

        # ---------------- PORTFOLIO ALLOCATION ----------------
        sec("ðŸª™","AI Portfolio Allocation")
        # Simple heuristic allocation based on risk level
        if risk_level_sel == "Low":
            alloc = {"BTC":50,"ETH":30,"SOL":12,"ADA":8}
        elif risk_level_sel == "Medium":
            alloc = {"BTC":35,"ETH":30,"SOL":22,"ADA":13}
        else:
            alloc = {"BTC":20,"ETH":25,"SOL":35,"ADA":20}

        alloc_colors = {"BTC":"#f59e0b","ETH":"#818cf8","SOL":"#34d399","ADA":"#a78bfa"}
        alloc_html = ""
        for c, pct in alloc.items():
            alloc_amt = inv_amount * pct/100
            alloc_html += f"""
            <div class="alloc-row">
                <div class="alloc-coin">{c}</div>
                <div class="alloc-bar-bg"><div class="alloc-bar-fill" style="width:{pct}%;background:{alloc_colors[c]}"></div></div>
                <div class="alloc-pct">{pct}%</div>
                <div style="width:110px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:.82rem;color:#94a3b8">{format_currency(alloc_amt, DISPLAY_CURRENCY)}</div>
            </div>"""
        st.markdown(f'<div class="trade-summary"><div class="trade-summary-title">AI-Suggested Diversification ({risk_level_sel} Risk)</div>{alloc_html}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close invest-panel

    # RISK
    sec("âš ï¸","Risk Analysis")
    volatility    = df["price"].pct_change().std()
    ann_vol       = volatility*np.sqrt(365)*100
    daily_ret     = df["price"].pct_change().dropna()
    sharpe        = (daily_ret.mean()/daily_ret.std())*np.sqrt(365) if daily_ret.std()>0 else 0
    max_dd        = ((df["price"]/df["price"].cummax())-1).min()*100
    if   volatility<.02: risk_level,risk_cls = "LOW","risk-low"
    elif volatility<.05: risk_level,risk_cls = "MEDIUM","risk-med"
    else:                risk_level,risk_cls = "HIGH","risk-high"

    st.markdown(f'<div class="risk-row"><span style="color:#64748b;font-size:.85rem">Overall Risk:</span><span class="risk-badge {risk_cls}">{risk_level} RISK</span></div>', unsafe_allow_html=True)
    rr1,rr2,rr3,rr4 = st.columns(4)
    rr1.metric("Daily Volatility",  f"{volatility*100:.2f}%")
    rr2.metric("Annual Volatility", f"{ann_vol:.1f}%")
    rr3.metric("Sharpe Ratio",      f"{sharpe:.2f}")
    rr4.metric("Max Drawdown",      f"{max_dd:.1f}%")

    rfig = go.Figure()
    rfig.add_trace(go.Histogram(x=daily_ret*100,nbinsx=50,marker_color="#6366f1",opacity=.7,name="Daily Returns"))
    rfig.add_vline(x=0,line=dict(color="#f59e0b",width=1.5,dash="dash"))
    rfig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",height=230,
                       title=dict(text="Daily Return Distribution",font=dict(size=12,color="#64748b")),
                       xaxis=dict(title="Return %",gridcolor="#0f172a"),yaxis=dict(title="Frequency",gridcolor="#0f172a"),
                       showlegend=False,margin=dict(l=10,r=10,t=36,b=10))
    st.plotly_chart(rfig, use_container_width=True)

    # INSIGHTS
    sec("ðŸ“Œ","AI Market Insights")
    bb_pos  = "near upper band" if df["price"].iloc[-1]>=df["bb_upper"].iloc[-1]*.98 else "near lower band" if df["price"].iloc[-1]<=df["bb_lower"].iloc[-1]*1.02 else "within Bollinger Bands"
    macd_lbl= "MACD above signal - bullish" if df["macd"].iloc[-1]>df["macd_signal"].iloc[-1] else "MACD below signal - bearish"
    ma_lbl  = "above both MA20 & MA50 - uptrend" if current_price>df["ma20"].iloc[-1] and current_price>df["ma50"].iloc[-1] else "below both MAs - downtrend" if current_price<df["ma20"].iloc[-1] and current_price<df["ma50"].iloc[-1] else "between MA20 & MA50 - mixed"
    for icon,txt in [
        ("ðŸ“Š",f"<b>Trend:</b> {symbol} is {ma_lbl}."),
        ("ðŸ“‰",f"<b>Momentum (RSI {rsi_current:.1f}):</b> {rsi_label} - {'watch for reversal.' if rsi_current>65 or rsi_current<35 else 'no extreme signal.'}"),
        ("ðŸŽ¯",f"<b>Bollinger Bands:</b> Price is {bb_pos}."),
        ("âš¡",f"<b>MACD:</b> {macd_lbl}."),
        ("ðŸ›¡ï¸",f"<b>Risk:</b> {risk_level} ({ann_vol:.1f}% annual vol). Sharpe {sharpe:.2f}.")
    ]:
        st.markdown(f'<div class="insight-pill"><span class="ic">{icon}</span><span>{txt}</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="footer">Built by <span>Nagesh</span> | CryptoSage AI v3.0 | Model: {model_choice} | <span>Not financial advice</span></div>', unsafe_allow_html=True)


# ===================================================
# ============  VIEW: MANUAL MODEL  =================
# ===================================================
elif VIEW == "Manual Model":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">Zero API | Fully Local | No Internet Required</div>
        <p class="hero-title"><span>Manual Model</span> Lab</p>
        <p class="hero-sub">Build, train & test prediction models on synthetic or uploaded data - no API key needed</p>
    </div>""", unsafe_allow_html=True)

    # ---- TABS inside manual model ----
    tab1, tab2, tab3 = st.tabs(["Build & Train", "Visualise Results", "Predict"])

    # ---- shared state ----
    if "mm_trained" not in st.session_state:
        st.session_state.mm_trained      = False
        st.session_state.mm_model_type   = "Linear Regression"
        st.session_state.mm_results      = {}
        st.session_state.mm_df           = None

    # =====================  TAB 1: BUILD  =====================
    with tab1:
        st.markdown('<div class="mm-card"><div class="mm-title">Step 1 - Data Source</div><div class="mm-desc">Choose synthetic data (no internet) or upload your own CSV.</div>', unsafe_allow_html=True)

        data_src = st.radio("Data Source", ["Synthetic (Sine + Noise)", "Upload CSV"], horizontal=True)

        if data_src == "Synthetic (Sine + Noise)":
            c1,c2,c3 = st.columns(3)
            n_points  = c1.slider("Data Points",   100, 2000, 500)
            trend     = c2.selectbox("Trend",       ["Uptrend","Downtrend","Sideways","Volatile"])
            noise_lvl = c3.slider("Noise Level",    0.01, 0.5, 0.1, 0.01)
            base      = 50000.0
            t         = np.linspace(0, 4*np.pi, n_points)
            if   trend=="Uptrend":   prices_raw = base + base*0.4*(t/(4*np.pi)) + base*0.08*np.sin(t*3)
            elif trend=="Downtrend": prices_raw = base + base*0.4*(1 - t/(4*np.pi)) + base*0.08*np.sin(t*3)
            elif trend=="Sideways":  prices_raw = base + base*0.1*np.sin(t*2)
            else:                    prices_raw = base + base*0.15*np.sin(t*5)*np.cos(t*2)
            prices_raw += prices_raw * noise_lvl * np.random.randn(n_points)
            prices_raw  = np.abs(prices_raw)
            dates        = pd.date_range(end=pd.Timestamp.today(), periods=n_points, freq="h")
            mm_df        = pd.DataFrame({"price": prices_raw}, index=dates)
            mm_df        = add_dual_currency_columns(mm_df, "INR")
            mm_df        = set_active_currency(mm_df, DISPLAY_CURRENCY)
            st.session_state.mm_df = mm_df
            st.success(f"Generated {n_points} synthetic data points | {trend}")
        else:
            uploaded = st.file_uploader("Upload CSV (must have a 'price' column)", type=["csv"])
            if uploaded:
                mm_df = pd.read_csv(uploaded)
                mm_df.columns = [col.strip().lower() for col in mm_df.columns]
                currency = st.selectbox("CSV Currency", ["USD", "INR"])

                if "date" in mm_df.columns:
                    mm_df["date"] = pd.to_datetime(mm_df["date"], errors="coerce")
                    mm_df = mm_df.dropna(subset=["date"]).sort_values("date").set_index("date")

                price_col = next((c for c in ["price", "close", "adj close"] if c in mm_df.columns), None)
                if price_col is None:
                    st.error("CSV must contain Price / Close column")
                else:
                    mm_df.rename(columns={price_col: "price"}, inplace=True)
                    mm_df["price"] = pd.to_numeric(mm_df["price"], errors="coerce")
                    mm_df = mm_df.dropna(subset=["price"])
                    if not isinstance(mm_df.index, pd.DatetimeIndex):
                        mm_df.index = pd.RangeIndex(len(mm_df))
                    mm_df = add_dual_currency_columns(mm_df, currency)
                    mm_df = set_active_currency(mm_df, DISPLAY_CURRENCY)
                    st.session_state.mm_df = mm_df
                    st.session_state.mm_currency = DISPLAY_CURRENCY
                    st.success(f"Loaded {len(mm_df)} rows | Current: {both_currency_text(mm_df['price'].iloc[-1], DISPLAY_CURRENCY)}")
            mm_df = st.session_state.mm_df

        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Model config ----
        st.markdown('<div class="mm-card"><div class="mm-title">Step 2 - Model Configuration</div><div class="mm-desc">All parameters are manual - no external libraries needed beyond scikit-learn and keras.</div>', unsafe_allow_html=True)

        m1,m2 = st.columns(2)
        mm_model_type = m1.selectbox(
            "Model Type",
            ["Linear Regression","LSTM Neural Net","Simple Moving Average (SMA)","Exponential Smoothing (EMA)","Polynomial Regression"]
        )
        window_mm = m2.slider("Lookback Window", 5, 60, 15)

        if mm_model_type == "LSTM Neural Net":
            l1,l2,l3,l4 = st.columns(4)
            lstm_units1 = l1.slider("LSTM Units Layer 1", 16, 128, 64, 16)
            lstm_units2 = l2.slider("LSTM Units Layer 2", 8,  64,  32, 8)
            dropout_r   = l3.slider("Dropout Rate",       0.0, 0.5, 0.2, 0.05)
            epochs_mm   = l4.slider("Epochs",             3, 30, 10)
        elif mm_model_type == "Polynomial Regression":
            poly_deg = st.slider("Polynomial Degree", 2, 5, 2)

        train_split = st.slider("Train / Test Split %", 60, 90, 80)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- TRAIN button ----
        if st.button("Train Model", use_container_width=True, type="primary"):
            mm_df = st.session_state.mm_df
            if mm_df is None:
                st.error("Please select or upload data first.")
            else:
                with st.spinner("Training model on your data..."):
                    prices_mm  = mm_df["price"].values.reshape(-1,1)
                    sc_mm      = MinMaxScaler()
                    scaled_mm  = sc_mm.fit_transform(prices_mm)

                    X_mm,y_mm = [],[]
                    for i in range(window_mm,len(scaled_mm)):
                        X_mm.append(scaled_mm[i-window_mm:i]); y_mm.append(scaled_mm[i])
                    X_mm = np.array(X_mm); y_mm = np.array(y_mm)

                    sp       = int(len(X_mm)*train_split/100)
                    Xtr,Xte  = X_mm[:sp],X_mm[sp:]
                    ytr,yte  = y_mm[:sp],y_mm[sp:]

                    # ---- choose model ----
                    trained_model = None
                    if mm_model_type == "Linear Regression":
                        trained_model = LinearRegression()
                        trained_model.fit(Xtr.reshape(Xtr.shape[0],-1), ytr)
                        preds_sc = trained_model.predict(Xte.reshape(Xte.shape[0],-1)).flatten()

                    elif mm_model_type == "Polynomial Regression":
                        from sklearn.preprocessing import PolynomialFeatures
                        from sklearn.pipeline import Pipeline
                        trained_model = Pipeline([
                            ("poly", PolynomialFeatures(degree=poly_deg)),
                            ("lr",   LinearRegression())
                        ])
                        trained_model.fit(Xtr.reshape(Xtr.shape[0],-1), ytr.flatten())
                        preds_sc = trained_model.predict(Xte.reshape(Xte.shape[0],-1)).flatten()

                    elif mm_model_type == "LSTM Neural Net":
                        trained_model = Sequential([
                            LSTM(lstm_units1,return_sequences=True,input_shape=(window_mm,1)),
                            Dropout(dropout_r),
                            LSTM(lstm_units2),
                            Dropout(dropout_r),
                            Dense(16), Dense(1)
                        ])
                        trained_model.compile(optimizer="adam",loss="huber")
                        trained_model.fit(Xtr,ytr,epochs=epochs_mm,batch_size=32,verbose=0)
                        preds_sc = trained_model.predict(Xte,verbose=0).flatten()

                    elif mm_model_type == "Simple Moving Average (SMA)":
                        # Pure numpy SMA - zero external ML
                        sma_raw  = pd.Series(prices_mm.flatten()).rolling(window_mm).mean().values
                        sma_sc   = sc_mm.transform(sma_raw.reshape(-1,1)).flatten()
                        preds_sc = sma_sc[window_mm+sp-1:window_mm+sp-1+len(yte)]
                        preds_sc = preds_sc[:len(yte)]
                        trained_model = "SMA"

                    elif mm_model_type == "Exponential Smoothing (EMA)":
                        alpha     = 0.3
                        ema_vals  = [prices_mm[0][0]]
                        for v in prices_mm[1:]:
                            ema_vals.append(alpha*v[0]+(1-alpha)*ema_vals[-1])
                        ema_sc   = sc_mm.transform(np.array(ema_vals).reshape(-1,1)).flatten()
                        preds_sc = ema_sc[window_mm+sp:window_mm+sp+len(yte)]
                        preds_sc = preds_sc[:len(yte)]
                        trained_model = "EMA"

                    # pad if needed
                    min_len  = min(len(preds_sc),len(yte))
                    preds_sc = preds_sc[:min_len]
                    yte_use  = yte[:min_len]

                    preds_true = sc_mm.inverse_transform(preds_sc.reshape(-1,1)).flatten()
                    y_true_mm  = sc_mm.inverse_transform(yte_use.reshape(-1,1)).flatten()

                    mae_mm  = mean_absolute_error(y_true_mm,preds_true)
                    rmse_mm = np.sqrt(mean_squared_error(y_true_mm,preds_true))
                    mape_mm = np.mean(np.abs((y_true_mm-preds_true)/np.maximum(y_true_mm,1e-8)))*100
                    acc_mm  = max(0,100-mape_mm)

                    # Next-step prediction
                    # ===========================

                    last_w_mm = scaled_mm[-window_mm:]

                    # Linear / Polynomial Regression
                    if mm_model_type in ["Linear Regression", "Polynomial Regression"]:

                        next_sc = trained_model.predict(
                            last_w_mm.reshape(1, -1)
                        )

                        next_sc = float(
                            np.array(next_sc).flatten()[0]
                        )

                    # LSTM
                    elif mm_model_type == "LSTM Neural Net":

                        next_sc = trained_model.predict(
                            last_w_mm.reshape(1, window_mm, 1),
                            verbose=0
                        )

                        next_sc = float(
                            np.array(next_sc).flatten()[0]
                        )

                    # Simple Moving Average
                    elif mm_model_type == "Simple Moving Average (SMA)":

                        next_sc = float(
                            np.mean(
                                scaled_mm[-window_mm:]
                            )
                        )

                    # Exponential Moving Average
                    else:

                        alpha = 0.3
                        ev    = scaled_mm[-window_mm][0]

                        for v in scaled_mm[-window_mm + 1:]:
                            ev = alpha * v[0] + (1 - alpha) * ev

                        next_sc = float(ev)

                    # Final inverse scaling (SAFE)
                    next_price = sc_mm.inverse_transform(
                        np.array([[next_sc]])
                    )[0][0]

                    # last_w_mm = scaled_mm[-window_mm:]
                    # if mm_model_type in ["Linear Regression","Polynomial Regression"]:
                    #     next_sc = trained_model.predict(last_w_mm.reshape(1,-1))[0]
                    # elif mm_model_type == "LSTM Neural Net":
                    #     next_sc = trained_model.predict(last_w_mm.reshape(1,window_mm,1),verbose=0)[0][0]
                    # elif mm_model_type == "Simple Moving Average (SMA)":
                    #     next_sc = float(np.mean(scaled_mm[-window_mm:]))
                    # else:  # EMA
                    #     alpha   = 0.3
                    #     ev      = scaled_mm[-window_mm][0]
                    #     for v in scaled_mm[-window_mm+1:]:
                    #         ev = alpha*v[0]+(1-alpha)*ev
                    #     next_sc = float(ev)

                    # next_price = sc_mm.inverse_transform([[next_sc]])[0][0]         



                    st.session_state.mm_trained    = True
                    st.session_state.mm_model_type = mm_model_type
                    st.session_state.mm_results    = {
                        "mae":mae_mm,"rmse":rmse_mm,"mape":mape_mm,"acc":acc_mm,
                        "preds":preds_true,"y_true":y_true_mm,
                        "next_price":next_price,"current_price":mm_df["price"].iloc[-1],"currency":DISPLAY_CURRENCY,
                        "dates":mm_df.index[window_mm+sp:window_mm+sp+min_len],
                        "full_df":mm_df
                    }
                    st.success(f"Model {mm_model_type} trained! Accuracy: {acc_mm:.1f}%")

    # =====================  TAB 2: VISUALISE  =====================
    with tab2:
        if not st.session_state.mm_trained:
            st.info("Train a model in the **Build & Train** tab first.")
        else:
            res     = st.session_state.mm_results
            mm_type = st.session_state.mm_model_type

            # Accuracy KPIs
            sec("ðŸ“‹","Model Performance")
            kc = (
                kpi_html("Accuracy",  f"{res['acc']:.1f}%",  None, "linear-gradient(90deg,#34d399,#6ee7b7)") +
                kpi_html("MAE",       format_currency(res['mae'], res.get("currency", DISPLAY_CURRENCY)), None, "linear-gradient(90deg,#6366f1,#818cf8)") +
                kpi_html("RMSE",      format_currency(res['rmse'], res.get("currency", DISPLAY_CURRENCY)),None, "linear-gradient(90deg,#f59e0b,#fbbf24)") +
                kpi_html("MAPE",      f"{res['mape']:.2f}%", None, "linear-gradient(90deg,#a78bfa,#c4b5fd)")
            )
            st.markdown(f'<div class="kpi-grid">{kc}</div>', unsafe_allow_html=True)

            # Full history chart
            sec("ðŸ“ˆ","Full Price History (Synthetic / Uploaded)")
            full_fig = go.Figure()
            full_fig.add_trace(go.Scatter(
                x=res["full_df"].index, y=res["full_df"]["price"],
                mode="lines", line=dict(color="#f59e0b",width=1.8),
                fill="tozeroy", fillcolor="rgba(245,158,11,.05)", name="Price"
            ))
            full_fig.update_layout(
                template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",
                height=280,font=dict(family="Space Grotesk",color="#64748b",size=11),
                xaxis=dict(gridcolor="#0f172a"),yaxis=dict(gridcolor="#0f172a",tickprefix=currency_tick_prefix(DISPLAY_CURRENCY),tickformat=",.0f"),
                showlegend=False,margin=dict(l=10,r=10,t=10,b=10)
            )
            st.plotly_chart(full_fig, use_container_width=True)

            # Actual vs Predicted
            sec("ðŸ”¬",f"Actual vs Predicted - {mm_type}")
            cmp_fig = go.Figure()
            cmp_fig.add_trace(go.Scatter(x=res["dates"],y=res["y_true"],line=dict(color="#f59e0b",width=2),name="Actual"))
            cmp_fig.add_trace(go.Scatter(x=res["dates"],y=res["preds"], line=dict(color="#6366f1",width=1.8,dash="dash"),name="Predicted"))
            cmp_fig.update_layout(
                template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",
                height=320,font=dict(family="Space Grotesk",color="#64748b",size=11),
                legend=dict(orientation="h",yanchor="bottom",y=1.01,x=0,bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="#0f172a"),yaxis=dict(gridcolor="#0f172a",tickprefix=currency_tick_prefix(DISPLAY_CURRENCY),tickformat=",.0f"),
                margin=dict(l=10,r=10,t=10,b=10)
            )
            st.plotly_chart(cmp_fig, use_container_width=True)

            # Error distribution
            sec("ðŸ“‰","Prediction Error Distribution")
            errors = res["y_true"] - res["preds"]
            err_fig = go.Figure()
            err_fig.add_trace(go.Histogram(x=errors,nbinsx=40,marker_color="#6366f1",opacity=.75,name="Error"))
            err_fig.add_vline(x=0,line=dict(color="#f59e0b",width=1.5,dash="dash"))
            err_fig.update_layout(
                template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",
                height=220,showlegend=False,
                xaxis=dict(title="Error",gridcolor="#0f172a"),
                yaxis=dict(title="Frequency",gridcolor="#0f172a"),
                margin=dict(l=10,r=10,t=10,b=10)
            )
            st.plotly_chart(err_fig, use_container_width=True)

    # =====================  TAB 3: PREDICT  =====================
    with tab3:
        if not st.session_state.mm_trained:
            st.info("Train a model in the **Build & Train** tab first.")
        else:
            res     = st.session_state.mm_results
            mm_type = st.session_state.mm_model_type
            cp      = res["current_price"]
            np_     = res["next_price"]
            chg     = ((np_-cp)/cp)*100

            sec("ðŸ”®","Next Step Prediction")
            if   np_ > cp*1.02: sc2,si2,sg2 = "buy","","BUY"
            elif np_ < cp*0.98: sc2,si2,sg2 = "sell","","SELL"
            else:               sc2,si2,sg2 = "hold","","HOLD"

            st.markdown(f"""
            <div class="signal-banner {sc2}">
                <div class="sig-icon">{si2}</div>
                <div>
                    <div class="sig-title {sc2}">{sg2} Signal | {mm_type}</div>
                    <div class="sig-desc">Trained on local data - no API key used. Next predicted value shown on right.</div>
                </div>
                <div style="margin-left:auto;text-align:right">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;color:#f1f5f9;font-weight:700">{both_currency_text(np_, res.get("currency", DISPLAY_CURRENCY))}</div>
                    <div style="font-size:.73rem;color:#475569">Predicted Next Value</div>
                </div>
            </div>""", unsafe_allow_html=True)

            col_a,col_b,col_c = st.columns(3)
            col_a.metric("Current Value",   both_currency_text(cp, res.get("currency", DISPLAY_CURRENCY)))
            col_b.metric("Predicted Value", both_currency_text(np_, res.get("currency", DISPLAY_CURRENCY)))
            col_c.metric("Expected Change", f"{chg:+.2f}%")

            # Accuracy summary
            sec("ðŸ“‹","Model Summary")
            st.markdown(f"""
            <div class="mm-card">
                <div class="mm-title">{mm_type}</div>
                <div class="mm-desc">Trained entirely locally | zero external APIs</div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:8px">
                    <div><div class="kpi-label">Accuracy</div><div class="kpi-value" style="color:#34d399">{res['acc']:.1f}%</div></div>
                    <div><div class="kpi-label">MAE</div><div class="kpi-value">{format_currency(res['mae'], res.get("currency", DISPLAY_CURRENCY))}</div></div>
                    <div><div class="kpi-label">RMSE</div><div class="kpi-value">{format_currency(res['rmse'], res.get("currency", DISPLAY_CURRENCY))}</div></div>
                    <div><div class="kpi-label">MAPE</div><div class="kpi-value">{res['mape']:.2f}%</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="footer">Built by <span>Nagesh</span> | CryptoSage AI v3.0 | Manual Model runs 100% locally | <span>Not financial advice</span></div>', unsafe_allow_html=True)






