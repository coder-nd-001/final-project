import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Premium Crypto Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= PREMIUM BLACK & GOLD THEME =================
st.markdown("""
<style>

/* ---------- MAIN APP ---------- */
.stApp {
    background:
    linear-gradient(rgba(0,0,0,0.82),
    rgba(0,0,0,0.82)),
    url("https://images.unsplash.com/photo-1640161704729-cbe966a08476?auto=format&fit=crop&w=1920&q=80");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(
    180deg,
    #0F0F0F,
    #1A1A1A
    );
    border-right: 2px solid #FFD700;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #FFD700 !important;
}

/* ---------- TEXT ---------- */
h1, h2, h3, h4, p, label {
    color: white !important;
}

/* ---------- METRIC CARDS ---------- */
.metric-box {
    background: rgba(255,215,0,0.08);
    border: 1px solid rgba(255,215,0,0.3);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
}

/* ---------- BUTTON ---------- */
.stButton>button {
    background-color: #FFD700;
    color: black;
    border-radius: 12px;
    border: none;
    font-weight: bold;
}

/* ---------- INPUT BOX ---------- */
.stNumberInput input,
.stSelectbox div {
    background-color: #111111 !important;
    color: white !important;
    border-radius: 10px !important;
}

/* ---------- GOLD LINE ---------- */
hr {
    border: 1px solid #FFD700;
}

/* ---------- CARD ---------- */
.crypto-card {
    background: rgba(255,215,0,0.07);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,215,0,0.2);
    backdrop-filter: blur(10px);
}

/* ---------- TITLE ---------- */
.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #FFD700;
    text-align: center;
}

/* ---------- SUBTITLE ---------- */
.subtitle {
    text-align:center;
    font-size:18px;
    color:#E5E5E5;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown(
    "<p class='big-title'>₿ Premium Crypto Price Predictor</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>AI Powered Cryptocurrency Forecasting using LSTM + Linear Regression</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Dashboard Settings")

coin = st.sidebar.selectbox(
    "Select Cryptocurrency",
    [
        "bitcoin",
        "ethereum",
        "litecoin"
    ]
)

investment = st.sidebar.number_input(
    "Investment Amount ₹",
    min_value=1000,
    value=10000,
    step=1000
)

days = st.sidebar.slider(
    "Historical Days",
    min_value=30,
    max_value=365,
    value=180
)

st.sidebar.markdown("---")
st.sidebar.success("✔ Premium Crypto Dashboard")
st.sidebar.info("Powered by AI Prediction")

# ================= FETCH DATA =================
@st.cache_data
def fetch_crypto_data(coin, days):

    url = (
        f"https://api.coingecko.com/api/v3/"
        f"coins/{coin}/market_chart"
    )

    params = {
        "vs_currency": "inr",
        "days": days
    }

    response = requests.get(
        url,
        params=params
    )

    data = response.json()

    prices = data["prices"]

    df = pd.DataFrame(
        prices,
        columns=["timestamp", "price"]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms"
    )

    df.set_index(
        "timestamp",
        inplace=True
    )

    return df


with st.spinner("📡 Fetching Live Crypto Data..."):
    df = fetch_crypto_data(
        coin,
        days
    )

# ================= CHART =================
st.subheader(
    f"📈 {coin.upper()} Market Trend"
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["price"],
        mode="lines",
        line=dict(
            color="gold",
            width=3
        ),
        name="Price"
    )
)

fig.update_layout(
    template="plotly_dark",
    height=500,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    xaxis_title="Date",
    yaxis_title="Price ₹"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ================= PREPROCESS =================
prices = df["price"].values.reshape(-1, 1)

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    prices
)

X = []
y = []

window_size = 10

for i in range(
    window_size,
    len(scaled_data)
):
    X.append(
        scaled_data[
            i-window_size:i
        ]
    )

    y.append(
        scaled_data[i]
    )

X = np.array(X)
y = np.array(y)

# ================= TRAIN MODELS =================
X_lr = X.reshape(
    X.shape[0],
    -1
)

lr_model = LinearRegression()
lr_model.fit(X_lr, y)

model = Sequential([
    LSTM(
        50,
        return_sequences=True,
        input_shape=(
            X.shape[1],
            1
        )
    ),
    LSTM(50),
    Dense(25),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

with st.spinner(
    "🤖 Training AI Model..."
):
    model.fit(
        X,
        y,
        epochs=5,
        batch_size=32,
        verbose=0
    )

# ================= PREDICTION =================
last_window = scaled_data[-window_size:]

X_pred = np.array(
    [last_window]
)

lr_pred = lr_model.predict(
    X_pred.reshape(1, -1)
)[0][0]

lstm_pred = model.predict(
    X_pred,
    verbose=0
)[0][0]

pred_scaled = (
    lr_pred + lstm_pred
) / 2

predicted_price = scaler.inverse_transform(
    [[pred_scaled]]
)[0][0]

current_price = (
    df["price"]
    .iloc[-1]
)

change_percent = (
    (
        predicted_price
        - current_price
    ) / current_price
) * 100

# ================= METRICS =================
st.subheader("📊 Market Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "💰 Current Price",
    f"₹{current_price:,.2f}"
)

col2.metric(
    "🔮 Predicted Price",
    f"₹{predicted_price:,.2f}"
)

col3.metric(
    "📈 Expected Change",
    f"{change_percent:.2f}%"
)

# ================= BUY / SELL =================
st.subheader("📢 Trading Recommendation")

if predicted_price > current_price * 1.02:
    st.success("📈 BUY Signal")

elif predicted_price < current_price * 0.98:
    st.error("📉 SELL Signal")

else:
    st.warning("⚖️ HOLD Signal")

# ================= PORTFOLIO =================
st.subheader("💼 Portfolio Simulation")

coins_bought = (
    investment
    / current_price
)

future_value = (
    coins_bought
    * predicted_price
)

profit = (
    future_value
    - investment
)

c1, c2 = st.columns(2)

c1.metric(
    "Future Value",
    f"₹{future_value:,.2f}"
)

c2.metric(
    "Profit / Loss",
    f"₹{profit:,.2f}"
)

# ================= RISK POPUP =================
st.subheader("⚠️ Risk Analysis")

volatility = (
    df["price"]
    .pct_change()
    .std()
)

if volatility < 0.02:
    risk = "LOW"
    st.success(
        "🟢 Low Risk Investment"
    )
    st.toast(
        "🟢 LOW RISK MARKET",
        icon="✅"
    )

elif volatility < 0.05:
    risk = "MEDIUM"
    st.warning(
        "🟡 Medium Risk Investment"
    )
    st.toast(
        "🟡 MEDIUM RISK MARKET",
        icon="⚠️"
    )

else:
    risk = "HIGH"
    st.error(
        "🔴 High Risk Investment"
    )
    st.toast(
        "🔴 HIGH RISK MARKET",
        icon="🚨"
    )

# ================= MARKET INSIGHTS =================
st.subheader("📌 AI Insights")

st.info(f"""
### {coin.upper()} Analysis

✅ Hybrid AI Prediction Model

✅ Live Crypto Data (CoinGecko API)

✅ Buy / Sell Recommendation

✅ Portfolio Simulation

✅ Risk Detection System

Current Risk Level:
**{risk}**
""")

st.markdown("---")

st.caption(
    "🚀 Built by Nagesh | Premium Crypto Prediction System"
)