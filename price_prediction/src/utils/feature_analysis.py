import pandas as pd
import numpy as np
# import talib
from scipy.stats import linregress


def moving_averages(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Add moving averages to DataFrame.
    """

    for window in windows:
        df[f'ma_{window}'] = df['close_price'].rolling(window=window).mean()

    return df


def volatility(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add volatility to DataFrame.
    """

    df['volatility'] = df['close_price'].rolling(window=window).std()
    return df


def add_lags(df: pd.DataFrame, lags: list) -> pd.DataFrame:
    """
    Add lags to DataFrame.
    """

    for lag in lags:
        df[f'lag_{lag}'] = df['close_price'].shift(lag)

    return df


def add_day_period_features(df: pd.DataFrame, day_of_week: bool = True, day_of_month: bool = True, quarter: bool = True) -> pd.DataFrame:
    """
    Add day period features to DataFrame.
    """
    if day_of_week:
        df['day_of_week'] = df['open_time'].dt.dayofweek
    if day_of_month:
        df['day_of_month'] = df['open_time'].dt.day
    if quarter:
        df['quarter'] = df['open_time'].dt.quarter
    return df


# def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
#     """
#     Add ADX to DataFrame.
#     ADX is a trend indicator that measures the strength of a trend.
#     """

#     df['ADX'] = talib.ADX(df['high_price'], df['low_price'],
#                           df['close_price'], timeperiod=window)

#     return df


# def add_macd(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add MACD to DataFrame.
#     MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
#     """
#     df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
#         df['close_price'])
#     return df


def linear_trend_slope(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Add linear trend slope to DataFrame.
    Linear trend slope is a trend indicator that measures the slope of the linear regression line.
    """

    def slope(series):
        return linregress(range(window), series[-window:])[0]

    for window in windows:
        df[f'linear_trend_{window}'] = df['close_price'].rolling(
            window=window).apply(slope)

    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical features to DataFrame.
    Cyclical features are features that have a cyclical nature, such as months and days.
    """
    if 'close_month' not in df.columns:
        df['close_month'] = df['open_time'].dt.month
    if 'close_day' not in df.columns:
        df['close_day'] = df['open_time'].dt.day

    df['month_sin'] = np.sin(df['close_month'] * (2 * np.pi / 12))
    df['month_cos'] = np.cos(df['close_month'] * (2 * np.pi / 12))
    df['day_sin'] = np.sin(df['close_day'] * (2 * np.pi / 31))
    df['day_cos'] = np.cos(df['close_day'] * (2 * np.pi / 31))

    return df
