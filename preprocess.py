import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_clean(filepath):
    df = pd.read_csv(r"C:\Users\Praveen\Downloads\supermarket_forecasting\data\supermarket_sales.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    # Aggregate daily sales
    daily_sales = df.groupby('Date')['Total'].sum().reset_index()
    daily_sales = daily_sales.sort_values('Date')
    return daily_sales

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
