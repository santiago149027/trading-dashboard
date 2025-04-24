import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator, CCIIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# Cargar el modelo
best_model = joblib.load("modelo_nvda.pkl")

# Lista de acciones: Magnificent 7
tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA"]

# Indicadores requeridos
features = [
    "RSI", "MACD", "MACD_Signal", "SMA_10", "EMA_10", "Momentum", "Volume",
    "bb_bbm", "bb_bbh", "bb_bbl", "bb_bandwidth",
    "atr", "cci", "adx", "roc"
]

# Función para preparar datos
def preparar_datos(ticker):
    df = yf.download(ticker, start="2018-01-01", end=datetime.today().strftime('%Y-%m-%d'), group_by='column')

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or "Close" not in df.columns:
        return None

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    df["RSI"] = RSIIndicator(close=close).rsi()
    macd = MACD(close=close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["SMA_10"] = SMAIndicator(close=close, window=10).sma_indicator()
    df["EMA_10"] = EMAIndicator(close=close, window=10).ema_indicator()
    df["Momentum"] = close.diff(4)

    bb = BollingerBands(close=close)
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()
    df["bb_bandwidth"] = df["bb_bbh"] - df["bb_bbl"]

    atr = AverageTrueRange(high=high, low=low, close=close)
    df["atr"] = atr.average_true_range()

    cci = CCIIndicator(high=high, low=low, close=close)
    df["cci"] = cci.cci()

    adx = ADXIndicator(high=high, low=low, close=close)
    df["adx"] = adx.adx()

    roc = ROCIndicator(close=close)
    df["roc"] = roc.roc()

    df.columns = [str(c).strip() for c in df.columns]

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features].fillna(0)
    return df

# Interfaz de Streamlit
st.set_page_config(page_title="Señales Magnificent 7", layout="wide")
st.title("📊 Dashboard de Señales de Compra: Magnificent 7")
st.caption("Modelo entrenado con NVDA")

# Aplicar modelo
resultados = []

for ticker in tickers:
    df = preparar_datos(ticker)

    if df is None or len(df) < 30:
        resultados.append({"Ticker": ticker, "Recomendación": "❌ Datos insuficientes"})
        continue

    try:
        latest_data = df.tail(1)
        pred = best_model.predict(latest_data)[0]
        señal = "📈 Comprar" if pred == 1 else "🚫 No comprar"
        resultados.append({"Ticker": ticker, "Recomendación": señal})
    except Exception as e:
        resultados.append({"Ticker": ticker, "Recomendación": f"⚠️ Error: {str(e)}"})

# Mostrar resultados
st.dataframe(pd.DataFrame(resultados).set_index("Ticker"))
