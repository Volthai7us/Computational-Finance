import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """
    Split DataFrame into train and test sets.
    """

    X = df.drop(['close_price'], axis=1)
    y = df['close_price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test


def scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale DataFrame.
    """

    scaler = StandardScaler()
    df[['open_price', 'high_price', 'low_price', 'close_price', 'volume']] = scaler.fit_transform(
        df[['open_price', 'high_price', 'low_price', 'close_price', 'volume']])

    return df


def drop_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing values.
    """

    df = df.dropna()
    df = df.reset_index(drop=True)

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in DataFrame.
    """

    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')

    return df


def eliminate_outliers(df: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    """
    Eliminate outliers from DataFrame using z-score.
    """

    z_scores = stats.zscore(
        df[['open_price', 'high_price', 'low_price', 'close_price', 'volume']])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)

    return df[filtered_entries]
