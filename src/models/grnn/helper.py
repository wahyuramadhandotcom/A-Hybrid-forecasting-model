import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def prepare_timeseries_supervised(df, target_col, n_lag, test_size=0.2):
    df = df.copy()
    for i in range(1, n_lag+1):
        df[f'lag_{i}'] = df[target_col].shift(i)
    df = df.dropna()

    X = df[[f'lag_{i}' for i in range(1, n_lag+1)]].values
    y = df[target_col].values

    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y
