import streamlit as st
from datetime import date

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import time

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    retries = 0
    while retries < 3:
        try:
            data = yf.download(ticker, START, TODAY)
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            st.warning(f"Error loading data: {str(e)}. Retrying in 5 seconds...")
            time.sleep(5)
            retries += 1
    st.error("Failed to load data. Please try again later.")
    return None

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
if data is not None:
    data_load_state.text('Loading data... done!')

    # Convert index to datetime and set timezone
    data.index = pd.DatetimeIndex(data.index).tz_localize("UTC").tz_convert("Asia/Kolkata")

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        plt.plot(data.index, data['Open'], label="stock_open")
        plt.plot(data.index, data['Close'], label="stock_close")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Time Series data')
        plt.legend()
        st.pyplot()

    plot_raw_data()

    # Perform stock forecasting with SVR
    df_train = data[['Date', 'Close']]
    df_train['Date'] = pd.to_datetime(df_train['Date'])  # Convert to datetime

    # Split the data into features (X) and target variable (y)
    X = df_train[['Date']]
    y = df_train['Close']

    # Train the SVR model
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model.fit(X, y)

    # Generate future dates for prediction
    future_dates = pd.date_range(start=df_train['Date'].iloc[-1], periods=period, closed='right')

    # Make predictions for future dates
    forecast_dates = pd.to_datetime(future_dates, unit='s')
    forecast = model.predict(future_dates.to_numpy().reshape(-1, 1))

    # Prepare forecast data for plotting
    forecast_data = pd.DataFrame({'Date': forecast_dates, 'Close': forecast})

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast_data.tail())

    st.write(f'Forecast plot for {n_years} years')
    plt.plot(df_train['Date'], df_train['Close'], label='Actual')
    plt.plot(forecast_dates, forecast, label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Forecast')
    plt.legend()
    st.pyplot()
