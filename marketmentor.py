import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st
from streamlit.components.v1 import html

# Custom CSS for background, animations, and style
def add_custom_css():
    st.markdown("""
        <style>
        .stApp {
            background-image: url("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.anytimeastro.com%2Fblog%2Fastrology%2Fstock-market-prediction%2F&psig=AOvVaw0vFiKRQV550LP33HCBbKaK&ust=1744964483746000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCOD_xrXR3owDFQAAAAAdAAAAABAh"); /* Replace with a valid direct image URL */
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: black;
        }
        .animated-title {
            animation: float 3s ease-in-out infinite;
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            color: #ffffff;
            padding: 10px 20px;
            border: 2px solid #ffffff;
            border-radius: 15px;
            text-shadow: 0 0 15px #00ffff;
            box-shadow: 0 0 15px #00ffff;
        }
        @keyframes float {
            0% { transform: translatey(0px); }
            50% { transform: translatey(-10px); }
            100% { transform: translatey(0px); }
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="animated-title">NSE Navigator 🚀</div>', unsafe_allow_html=True)

# --- Streamlit App Starts ---
add_custom_css()

def fetch_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="5y")
    return data[['Close']]

def preprocess_data(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_50'] = df['Close'].ewm(span=20, adjust=False).mean()
    df.dropna(inplace=True)
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    return scaled_data, scaler

def prepare_data(scaled_data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i])
        y.append(scaled_data[i,0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

stock_list = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
selected_stock = st.selectbox('Select the Stock:', stock_list)

st.write(f"Fetching data for {selected_stock}...")
data = fetch_data(selected_stock)
latest_price = data['Close'].iloc[-1]
st.write(f'Latest Price: ₹{latest_price:.2f}')

if st.button("Train and Predict"):
    st.write("Please Wait...")

    data = preprocess_data(data)
    scaled_data, scaler = normalize_data(data)
    X, y = prepare_data(scaled_data)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model((x_train.shape[1], x_train.shape[2]))
    model.fit(x_train, y_train, batch_size=32, epochs=5)

    st.write("Predicting prices for the next 10 days...")
    predictions = []
    input_sequence = scaled_data[-60:]

    for day in range(10):
        input_sequence = input_sequence.reshape(1, -1, 1)
        predicted_price = model.predict(input_sequence)[0][0]
        predictions.append(predicted_price)
        input_sequence = np.append(input_sequence[0][1:], [[predicted_price]], axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))[:, 0]
    days = pd.date_range(start=pd.Timestamp.now() + pd.DateOffset(1), periods=10).strftime('%Y-%m-%d').tolist()
    prediction_df = pd.DataFrame({'Date': days, 'Predicted Price': predictions})

    st.write("Predicted Price:")
    st.table(prediction_df)

    # Graph with three lines
    fig = go.Figure()

    # Predicted Prices
    fig.add_trace(go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Predicted Price'],
        mode='lines+markers',
        name='Predicted Prices',
        line=dict(color='blue')
    ))

    # SMA_50
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA_50'],
        mode='lines',
        name='SMA 50',
        line=dict(color='green')
    ))

    # EMA_50
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA_50'],
        mode='lines',
        name='EMA 50',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f"Stock Analysis for {selected_stock}",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        legend_title="Metrics",
        template="plotly_dark"
    )

    st.plotly_chart(fig)
