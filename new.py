import streamlit as st
from datetime import date

import yfinance as yf
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig, ax = plt.subplots()
    ax.plot(data['Date'], data['Open'], label="stock_open")
    ax.plot(data['Date'], data['Close'], label="stock_close")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Time Series data')
    ax.legend()
    st.pyplot(fig)

plot_raw_data()

# Perform stock forecasting with Support Vector Regression (SVR)
df_train = data[['Date', 'Close']]
df_train['Date'] = pd.to_datetime(df_train['Date'])  # Convert to pandas datetime format

# Split the data into features (X) and target variable (y)
X = df_train.index.values.reshape(-1, 1)
y = df_train['Close'].values

# Train the SVR model
model = SVR(kernel='rbf')
model.fit(X, y)

# Generate future dates for prediction
last_date = df_train['Date'].iloc[-1]
future_dates = pd.date_range(start=pd.to_datetime(last_date), periods=period+1, closed='right')

# Make predictions for future dates
future_dates_unix = (future_dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
future_dates_index = np.arange(len(df_train), len(df_train) + len(future_dates))
forecast = model.predict(future_dates_index.reshape(-1, 1))

# Prepare forecast data for plotting
forecast_data = pd.DataFrame({'Date': future_dates, 'Close': forecast})

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast_data.tail())

st.write(f'Forecast plot for {n_years} years')
fig, ax = plt.subplots()
ax.plot(df_train['Date'], df_train['Close'], label='Actual')
ax.plot(future_dates, forecast, label='Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Stock Price Forecast')
ax.legend()

# Format x-axis ticks as dates
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

st.pyplot(fig)
