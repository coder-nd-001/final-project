# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# import plotly.graph_objects as go
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM

# # ---------------- PAGE CONFIG ----------------
# st.set_page_config(
#     page_title="Crypto Price Predictor",
#     page_icon="🚀",
#     layout="wide"
# )

# st.title("🚀 Advanced Crypto Price Predictor")

# # ---------------- SIDEBAR ----------------
# st.sidebar.header("Settings")

# coin = st.sidebar.selectbox(
#     "Select Cryptocurrency",
#     ["bitcoin", "ethereum", "litecoin"]
# )

# investment = st.sidebar.number_input(
#     "Investment Amount ₹",
#     value=10000
# )

# days = st.sidebar.slider(
#     "Historical Days",
#     30,
#     365,
#     180
# )

# # ---------------- FETCH DATA ----------------
# @st.cache_data
# def fetch_crypto_data(coin):

#     url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"

#     params = {
#         "vs_currency": "inr",
#         "days": days
#     }

#     response = requests.get(url, params=params)
#     data = response.json()

#     prices = data["prices"]

#     df = pd.DataFrame(prices, columns=["timestamp", "price"])

#     df["timestamp"] = pd.to_datetime(
#         df["timestamp"],
#         unit="ms"
#     )

#     df.set_index("timestamp", inplace=True)

#     return df


# df = fetch_crypto_data(coin)

# # ---------------- CHART ----------------
# st.subheader(f"📈 {coin.upper()} Price Chart")

# fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=df.index,
#     y=df["price"],
#     mode="lines",
#     name="Price"
# ))

# fig.update_layout(
#     template="plotly_dark",
#     height=500,
#     xaxis_title="Date",
#     yaxis_title="Price ₹"
# )

# st.plotly_chart(fig, use_container_width=True)

# # ---------------- PREPROCESS ----------------
# prices = df["price"].values.reshape(-1, 1)

# scaler = MinMaxScaler()

# scaled_data = scaler.fit_transform(prices)

# X = []
# y = []

# window_size = 10

# for i in range(window_size, len(scaled_data)):
#     X.append(scaled_data[i-window_size:i])
#     y.append(scaled_data[i])

# X = np.array(X)
# y = np.array(y)

# # ---------------- LINEAR REGRESSION ----------------
# X_lr = X.reshape(X.shape[0], -1)

# lr_model = LinearRegression()
# lr_model.fit(X_lr, y)

# # ---------------- LSTM MODEL ----------------
# model = Sequential([
#     LSTM(50, return_sequences=True,
#          input_shape=(X.shape[1], 1)),
#     LSTM(50),
#     Dense(25),
#     Dense(1)
# ])

# model.compile(
#     optimizer="adam",
#     loss="mean_squared_error"
# )

# model.fit(
#     X,
#     y,
#     epochs=5,
#     batch_size=32,
#     verbose=0
# )

# # ---------------- PREDICTION ----------------
# last_window = scaled_data[-window_size:]

# X_pred = np.array([last_window])

# lr_pred = lr_model.predict(
#     X_pred.reshape(1, -1)
# )[0][0]

# lstm_pred = model.predict(
#     X_pred,
#     verbose=0
# )[0][0]

# # Hybrid Prediction
# pred_scaled = (lr_pred + lstm_pred) / 2

# predicted_price = scaler.inverse_transform(
#     [[pred_scaled]]
# )[0][0]

# current_price = df["price"].iloc[-1]

# # ---------------- METRICS ----------------
# col1, col2, col3 = st.columns(3)

# with col1:
#     st.metric(
#         "Current Price",
#         f"₹{current_price:,.2f}"
#     )

# with col2:
#     st.metric(
#         "Predicted Price",
#         f"₹{predicted_price:,.2f}"
#     )

# with col3:
#     percent_change = (
#         (predicted_price - current_price)
#         / current_price
#     ) * 100

#     st.metric(
#         "Expected Change",
#         f"{percent_change:.2f}%"
#     )

# # ---------------- SIGNAL ----------------
# st.subheader("📢 Trading Signal")

# if predicted_price > current_price * 1.02:
#     st.success("📈 BUY Signal")

# elif predicted_price < current_price * 0.98:
#     st.error("📉 SELL Signal")

# else:
#     st.warning("⚖️ HOLD Signal")

# # ---------------- PORTFOLIO ----------------
# st.subheader("💼 Portfolio Simulation")

# coins_bought = investment / current_price

# future_value = (
#     coins_bought * predicted_price
# )

# profit = future_value - investment

# st.write(
#     f"💰 Future Value: ₹{future_value:,.2f}"
# )

# st.write(
#     f"📊 Profit/Loss: ₹{profit:,.2f}"
# )

# # ---------------- RISK ANALYSIS ----------------
# st.subheader("⚠️ Risk Analysis")

# volatility = df["price"].pct_change().std()

# if volatility < 0.02:
#     risk = "Low"
#     st.success(f"Risk Level: {risk}")

# elif volatility < 0.05:
#     risk = "Medium"
#     st.warning(f"Risk Level: {risk}")

# else:
#     risk = "High"
#     st.error(f"Risk Level: {risk}")

# st.markdown("---")
# st.caption(
#     "Built with Streamlit + LSTM + Linear Regression"
# )





import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="🚀",
    layout="wide"
)

# ---------------- BITCOIN BACKGROUND ----------------
st.markdown("""
<style>

.stApp {
    background-image: url("https://images.unsplash.com/photo-1621761191319-c6fb62004040");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

[data-testid="stAppViewContainer"]{
    background-color: rgba(0,0,0,0.75);
}

[data-testid="stSidebar"]{
    background-color: rgba(0,0,0,0.85);
}

h1, h2, h3, h4, p, label, div {
    color: white !important;
}

.metric-box {
    background-color: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🚀 Advanced Crypto Price Predictor")
st.write(
    "Predict cryptocurrency prices using "
    "**Linear Regression + LSTM Hybrid Model**"
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

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
    value=10000
)

days = st.sidebar.slider(
    "Historical Data (Days)",
    min_value=30,
    max_value=365,
    value=180
)

# ---------------- FETCH CRYPTO DATA ----------------
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
        columns=[
            "timestamp",
            "price"
        ]
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

# ---------------- LOAD DATA ----------------
with st.spinner("Fetching Crypto Data..."):
    df = fetch_crypto_data(
        coin,
        days
    )

# ---------------- CHART ----------------
st.subheader(
    f"📈 {coin.upper()} Price Trend"
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["price"],
        mode="lines",
        name="Price"
    )
)

fig.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_title="Date",
    yaxis_title="Price (₹)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------- PREPROCESS ----------------
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

# ---------------- LINEAR REGRESSION ----------------
X_lr = X.reshape(
    X.shape[0],
    -1
)

lr_model = LinearRegression()

lr_model.fit(
    X_lr,
    y
)

# ---------------- LSTM MODEL ----------------
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
    "Training AI Model..."
):
    model.fit(
        X,
        y,
        epochs=5,
        batch_size=32,
        verbose=0
    )

# ---------------- PREDICTION ----------------
last_window = scaled_data[
    -window_size:
]

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

# Hybrid Prediction
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

percent_change = (
    (
        predicted_price
        - current_price
    )
    / current_price
) * 100

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "💰 Current Price",
        f"₹{current_price:,.2f}"
    )

with col2:
    st.metric(
        "🔮 Predicted Price",
        f"₹{predicted_price:,.2f}"
    )

with col3:
    st.metric(
        "📊 Expected Change",
        f"{percent_change:.2f}%"
    )

# ---------------- BUY / SELL ----------------
st.subheader(
    "📢 Trading Signal"
)

if predicted_price > (
    current_price * 1.02
):
    st.success(
        "📈 BUY Signal"
    )

elif predicted_price < (
    current_price * 0.98
):
    st.error(
        "📉 SELL Signal"
    )

else:
    st.warning(
        "⚖️ HOLD Signal"
    )

# ---------------- PORTFOLIO ----------------
st.subheader(
    "💼 Portfolio Simulation"
)

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

col4, col5 = st.columns(2)

with col4:
    st.metric(
        "Future Value",
        f"₹{future_value:,.2f}"
    )

with col5:
    st.metric(
        "Profit / Loss",
        f"₹{profit:,.2f}"
    )

# ---------------- RISK ANALYSIS ----------------
st.subheader(
    "⚠️ Risk Analysis"
)

volatility = (
    df["price"]
    .pct_change()
    .std()
)

if volatility < 0.02:
    risk = "Low"
    st.success(
        f"Risk Level: {risk}"
    )

elif volatility < 0.05:
    risk = "Medium"
    st.warning(
        f"Risk Level: {risk}"
    )

else:
    risk = "High"
    st.error(
        f"Risk Level: {risk}"
    )

# ---------------- MARKET INFO ----------------
st.subheader(
    "📌 Market Insights"
)

st.info(f"""
Selected Coin: **{coin.upper()}**

• Hybrid AI Model (LSTM + Linear Regression)

• Real-time CoinGecko Data

• Buy / Sell Recommendation

• Portfolio Simulation

• Volatility Risk Analysis
""")

# ---------------- FOOTER ----------------
st.markdown("---")

st.caption(
    "Built with ❤️ using "
    "Streamlit, TensorFlow & AI"
)