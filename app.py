import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="AI Equity Research Platform",
    layout="wide"
)

st.title("AI Equity Research Platform")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    value="AAPL"
).upper()


@st.cache_data(ttl=3600)
def load_data(symbol):
    try:
        data = yf.download(
            symbol,
            period="5y",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data

    except Exception:
        return pd.DataFrame()


data = load_data(ticker)

if data.empty:
    st.error(
        f"No data found for {ticker}. Try AAPL, MSFT, NVDA, RELIANCE.NS, INFY.NS or TCS.NS"
    )
    st.stop()

close = data["Close"].squeeze()

# =========================
# Indicators
# =========================

data["SMA20"] = SMAIndicator(
    close=close,
    window=20
).sma_indicator()

data["SMA50"] = SMAIndicator(
    close=close,
    window=50
).sma_indicator()

data["RSI"] = RSIIndicator(
    close=close,
    window=14
).rsi()

macd = MACD(close)

data["MACD"] = macd.macd()
data["MACD_SIGNAL"] = macd.macd_signal()

bb = BollingerBands(
    close=close,
    window=20,
    window_dev=2
)

data["BB_UPPER"] = bb.bollinger_hband()
data["BB_LOWER"] = bb.bollinger_lband()

# =========================
# Sidebar
# =========================

st.sidebar.header("Market Snapshot")

st.sidebar.metric(
    "Current Price",
    f"{float(close.iloc[-1]):.2f}"
)

st.sidebar.metric(
    "52W High",
    f"{float(close.tail(252).max()):.2f}"
)

st.sidebar.metric(
    "52W Low",
    f"{float(close.tail(252).min()):.2f}"
)

# =========================
# Tabs
# =========================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Price Analysis",
        "Technical Analysis",
        "Risk Metrics",
        "Forecast"
    ]
)

# =========================
# Price Analysis
# =========================

with tab1:

    st.subheader(f"{ticker} Price Chart")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=close,
            name="Close"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA20"],
            name="SMA20"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA50"],
            name="SMA50"
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================
# Technical Analysis
# =========================

with tab2:

    st.subheader("RSI")

    st.line_chart(
        data["RSI"]
    )

    latest_rsi = float(
        data["RSI"].dropna().iloc[-1]
    )

    st.metric(
        "Current RSI",
        round(latest_rsi, 2)
    )

    if latest_rsi > 70:
        st.warning("Overbought")

    elif latest_rsi < 30:
        st.success("Oversold")

    else:
        st.info("Neutral")

    st.divider()

    st.subheader("MACD")

    st.line_chart(
        pd.DataFrame({
            "MACD": data["MACD"],
            "Signal": data["MACD_SIGNAL"]
        })
    )

    st.divider()

    st.subheader("Bollinger Bands")

    st.line_chart(
        pd.DataFrame({
            "Close": close,
            "Upper": data["BB_UPPER"],
            "Lower": data["BB_LOWER"]
        })
    )

# =========================
# Risk Metrics
# =========================

with tab3:

    returns = close.pct_change().dropna()

    annual_volatility = (
        returns.std() * np.sqrt(252)
    ) * 100

    sharpe_ratio = (
        returns.mean() /
        returns.std()
    ) * np.sqrt(252)

    drawdown = (
        close / close.cummax()
    ) - 1

    max_drawdown = (
        drawdown.min()
    ) * 100

    total_return = (
        (
            close.iloc[-1]
            /
            close.iloc[0]
        ) - 1
    ) * 100

    risk_df = pd.DataFrame({
        "Metric": [
            "5 Year Return %",
            "Annual Volatility %",
            "Sharpe Ratio",
            "Maximum Drawdown %"
        ],
        "Value": [
            round(total_return, 2),
            round(annual_volatility, 2),
            round(sharpe_ratio, 2),
            round(max_drawdown, 2)
        ]
    })

    st.dataframe(
        risk_df,
        use_container_width=True
    )

# =========================
# Forecast
# =========================

with tab4:

    st.subheader("ARIMA Forecast")

    monthly_prices = close.resample("ME").last()

    try:

        model = ARIMA(
            monthly_prices,
            order=(2, 1, 2)
        )

        model_fit = model.fit()

        forecast = model_fit.forecast(
            steps=12
        )

        future_dates = pd.date_range(
            start=monthly_prices.index[-1] + pd.offsets.MonthEnd(1),
            periods=12,
            freq="ME"
        )

        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=monthly_prices.index,
                y=monthly_prices,
                name="Historical"
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=forecast,
                name="Forecast"
            )
        )

        forecast_fig.update_layout(
            template="plotly_white",
            height=600
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast Price": forecast
        })

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Forecast unavailable: {e}"
        )
