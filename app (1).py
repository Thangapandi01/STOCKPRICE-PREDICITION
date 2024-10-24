import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import yfinance as yf

# Title of the web app
st.title('Stock Price Prediction using LSTM')

# Sidebar for user input
st.sidebar.header('User Input')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2012-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2022-12-21'))
stock = st.sidebar.text_input('Stock Ticker', 'GOOG')
n_days_predict = st.sidebar.number_input('Days to Predict Ahead', min_value=1, max_value=30, value=7)
epochs = st.sidebar.slider('Number of Epochs for Training', min_value=1, max_value=100, value=5)

# Load the data
st.subheader(f'Historical Data for {stock}')
df = yf.download(stock, start=start_date, end=end_date)
st.write(df.tail())

# Plot closing price
st.subheader('Closing Price vs Time')
plt.figure(figsize=(10, 6))
plt.plot(df['Close'], label='Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)

# Moving Averages
st.subheader('100 Days & 200 Days Moving Average')
ma_100 = df['Close'].rolling(100).mean()
ma_200 = df['Close'].rolling(200).mean()

plt.figure(figsize=(10, 6))
plt.plot(ma_100, 'r', label='100 Days MA')
plt.plot(ma_200, 'b', label='200 Days MA')
plt.plot(df['Close'], 'g', label='Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)

# Prepare the data
st.subheader('Data Preparation & LSTM Model Training')

df.dropna(inplace=True)
data_train = pd.DataFrame(df['Close'][0:int(len(df) * 0.80)])
data_test = pd.DataFrame(df['Close'][int(len(df) * 0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)

# Creating x_train and y_train
x_train = []
y_train = []
for i in range(100, data_train_scaled.shape[0]):
    x_train.append(data_train_scaled[i-100:i])
    y_train.append(data_train_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=1)

# Prepare test data
past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)

input_data = scaler.transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
y_predicted = model.predict(x_test)
scaler_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor

# Plot predictions
st.subheader('Predicted vs Original Closing Price')
plt.figure(figsize=(10, 6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)

# Predict future prices for the next 'n_days_predict' days
st.subheader(f'Prediction for Next {n_days_predict} Days')

# Use the last 100 days of the actual data to make future predictions
last_100_days_data = input_data[-100:]
last_100_days_data = np.reshape(last_100_days_data, (1, last_100_days_data.shape[0], 1))

predicted_future = []
for _ in range(n_days_predict):
    prediction = model.predict(last_100_days_data)
    predicted_future.append(prediction[0][0])
    
    # Update the input for the next prediction
    next_input = np.append(last_100_days_data[:, 1:, :], [[prediction]], axis=1)
    last_100_days_data = next_input

# Convert predicted future prices back to the original scale
predicted_future = np.array(predicted_future) * scaler_factor

# Plot future predictions
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_days_predict + 1), predicted_future, 'r', label=f'Next {n_days_predict} Days Predicted Price')
plt.xlabel('Days Ahead')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)
