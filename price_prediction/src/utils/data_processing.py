import pandas as pd
from scipy import stats
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from csv file and convert to pandas DataFrame.
    """

    df = pd.read_csv(path, header=None)
    df.columns = ['open_time', 'open_price', 'high_price',
                  'low_price', 'close_price', 'volume', 'close_time']
    df['open_time'] = pd.to_datetime(df['open_time'] * 1000, unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    return df


def check_missing_values(df: pd.DataFrame) -> None:
    """
    Check missing values in DataFrame.
    """

    print(f"Missing values in DataFrame: {df.isnull().sum().sum()}")
    print(f"Missing values in each column: {df.isnull().sum()}")
    print(f"Missing values in each row: {df.isnull().sum(axis=1)}")


def hello_world_2() -> str:
    """
    Return a string.
    """

    return 'Hello World!'
