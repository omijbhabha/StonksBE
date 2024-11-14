from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import math
from keras.layers import Input, LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

@app.route('/stonks', methods=['POST'])
def submit_option():
    data = request.json
    option = data.get('option')

    # Download stock data and preprocess
    stock = yf.download(option, period="3y", actions=True)
    data = stock.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * 0.75)

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Prepare training data
    train_data = scaled_data[:training_data_len, :]
    x_train = []
    y_train = []
    for i in range(30, len(train_data)):  # Reduced look-back window
        x_train.append(train_data[i - 30:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Simplified LSTM model
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(20, return_sequences=False))  # Reduced LSTM units and layers
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=3, batch_size=1, verbose=0)  # Fewer epochs

    # Prepare test data and make predictions
    test_data = scaled_data[training_data_len - 30:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(30, len(test_data)):
        x_test.append(test_data[i - 30:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate average deviation
    rmse = np.sqrt(((predictions - y_test) ** 2).mean())
    average_close = data['Close'].mean()
    average_deviation = rmse / average_close * 100

    # Plot and encode plot as base64
    train = data.iloc[:training_data_len]
    valid = data.iloc[training_data_len:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Value', 'Predictions'], loc='lower right')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode('utf-8')

    return jsonify({
        'message': f"Model prediction complete for {option}",
        'averageDeviation': average_deviation,
        'plot': plot_data
    })

if __name__ == '__main__':
    app.run(debug=True)
