from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import yfinance as yf

app = Flask(__name__)


# Function to identify head and shoulders pattern
def find_head_and_shoulders(df, window=20):
    patterns = []
    for i in range(window, len(df) - window):
        left_shoulder = df['Close'][i - window:i]
        head = df['Close'][i:i + window // 2]
        right_shoulder = df['Close'][i + window // 2:i + window]

        if max(left_shoulder) < max(head) and max(right_shoulder) < max(head) and min(left_shoulder) > min(
                right_shoulder):
            patterns.append((i - window, i + window))

    return patterns


# Route to render the web page
@app.route('/', methods=['GET', 'POST'])
def index():
    company = request.form.get('company', 'AAPL')
    start_date = request.form.get('start_date', '2020-01-01')
    end_date = request.form.get('end_date', '2023-01-01')

    try:
        stock_data = yf.download(company, start=start_date, end=end_date)
        patterns = find_head_and_shoulders(stock_data)

        # Create the example Head & Shoulders pattern plot
        x = np.arange(50)
        shoulder_len = len(x) // 3
        example_pattern = np.concatenate([
            np.linspace(20, 25, shoulder_len),
            np.linspace(25, 40, shoulder_len),
            np.linspace(40, 25, shoulder_len),
            np.linspace(25, 20, shoulder_len)
        ])
        example_fig = go.Figure(
            go.Scatter(x=x[:len(example_pattern)], y=example_pattern, mode='lines', name='Example H&S'))
        example_fig.update_layout(title='Example of Head & Shoulders Pattern', xaxis_title='Time', yaxis_title='Price')

        # Final Comprehensive Graph
        final_fig = go.Figure()
        final_fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price',
                                       line=dict(color='blue')))

        for idx, (start, end) in enumerate(patterns):
            final_fig.add_trace(
                go.Scatter(x=stock_data.index[start:end], y=stock_data['Close'][start:end], mode='lines',
                           name=f'Pattern {idx + 1}', line=dict(color='red')))

            pattern_end_date = stock_data.index[end]
            stock_data['Date'] = pd.to_numeric(stock_data.index.map(pd.Timestamp.timestamp))
            X = stock_data[['Date']]
            y = stock_data['Close']

            X_train = X.loc[:pattern_end_date]
            y_train = y.loc[:pattern_end_date]
            model = LinearRegression()
            model.fit(X_train, y_train)

            future_dates = np.array([X_train.iloc[-1, 0] + i * 24 * 60 * 60 for i in range(1, 31)]).reshape(-1, 1)
            predicted_prices = model.predict(future_dates)
            future_dates = pd.to_datetime(future_dates.flatten(), unit='s')

            final_fig.add_trace(
                go.Scatter(x=future_dates, y=predicted_prices, mode='lines', name=f'Predicted Prices {idx + 1}',
                           line=dict(dash='dash', color='orange')))
            final_fig.add_trace(
                go.Scatter(x=stock_data.index[end:end + 30], y=stock_data['Close'].iloc[end:end + 30], mode='lines',
                           name=f'Actual Future Prices {idx + 1}', line=dict(dash='dash', color='green')))

        final_fig.update_layout(title='Complete Stock Data with Head & Shoulders Patterns and Predictions',
                                xaxis_title='Date', yaxis_title='Price')

        # Convert plots to HTML
        example_plot_html = example_fig.to_html(full_html=False)
        final_plot_html = final_fig.to_html(full_html=False)

        return render_template('index.html', example_plot=example_plot_html, final_plot=final_plot_html)

    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == '__main__':
    app.run(debug=True)
