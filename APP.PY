from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import math
from datetime import datetime

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'nonsense'

# Replace with your Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = 'O377W6RWRERIM2TJ'

# Directory to save plots
if not os.path.exists('static'):
    os.makedirs('static')

# Function to fetch stock data from Alpha Vantage API
def fetch_stock_data(symbol):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    
    try:
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        data = data.sort_index(ascending=True)
        data = data.tail(365)  # Use 1 year of data
        data.reset_index(inplace=True)
        data.rename(columns={
            'date': 'Date', 
            '1. open': 'Open', 
            '2. high': 'High', 
            '3. low': 'Low', 
            '4. close': 'Close', 
            '5. volume': 'Volume'
        }, inplace=True)
        
        # Save as CSV
        csv_filename = f"data/{symbol}.csv"
        if not os.path.exists("data"):
            os.makedirs("data")
        print("saving..")
        data.to_csv(csv_filename, index=False)
        print("didnot save")
        
        return data
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# ARIMA Model for Prediction
def arima_model(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Using only Close price for prediction
    close_prices = data['Close'].values

    # Train-Test Split (80% train, 20% test)
    train_size = int(len(close_prices) * 0.8)
    train, test = close_prices[:train_size], close_prices[train_size:]

    # ARIMA Model fitting
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    predictions = model_fit.forecast(steps=len(test))
    error = math.sqrt(mean_squared_error(test, predictions))

    # Plotting Actual vs Predicted Prices
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[train_size:], test, label='Actual Price', color='blue')
    plt.plot(data.index[train_size:], predictions, label='Predicted Price', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('ARIMA Model: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plot_filename = 'static/arima_plot.png'
    plt.savefig(plot_filename)
    plt.close()

    # Forecasting for next 7 days
    future_forecast = model_fit.forecast(steps=7)

    return predictions, future_forecast, error, plot_filename

# Route for Homepage (Index)
@app.route('/')
def index():
    return render_template('index.html')

# Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form.get('symbol').upper()

    if symbol:
        stock_data = fetch_stock_data(symbol)

        if stock_data is not None:
            predictions, future_forecast, error, plot_filename = arima_model(stock_data)
            print("reached here")

            return render_template('result.html', 
                                   symbol=symbol,
                                   predictions=predictions,
                                   future_forecast=future_forecast,
                                   error=error,
                                   plot_filename=plot_filename)
        else:
            flash('Invalid stock symbol or no data found. Please try again.', 'danger')
            return redirect(url_for('index'))

    flash('Please enter a stock symbol.', 'warning')
    return redirect(url_for('index'))

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)