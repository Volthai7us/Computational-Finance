import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd


def plot_close_price(df: pd.DataFrame) -> None:
    """
    Plot stock or cryptocurrency close price.
    """

    plt.figure(figsize=(15, 5))
    plt.plot(df['open_time'], df['close_price'])
    plt.show()


def plot_feature_histograms(df: pd.DataFrame) -> None:
    """
    Plot histograms of features in DataFrame.
    """

    features = ['open_price', 'high_price',
                'low_price', 'close_price', 'volume']

    plt.subplots(figsize=(20, 15))
    for i, feature in enumerate(features):
        plt.subplot(3, 2, i+1)
        sb.histplot(df[feature])
    plt.show()


def plot_price_by_time(df: pd.DataFrame) -> None:
    """
    Plot close price by time.   
    """

    groupings = {
        'close_day': (range(1, 32), None),
        'close_dayofweek': (range(1, 8), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
        'close_month': (range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']),
        'close_quarter': ([1, 2, 3, 4], None)
    }

    plt.subplots(4, 1, figsize=(20, 15))
    for i, (group, (ticks, labels)) in enumerate(groupings.items()):
        grouped = df.groupby(group).mean()
        plt.subplot(4, 1, i+1)
        plt.plot(grouped['close_price'])
        plt.xticks(ticks, labels)
        plt.title(f'Close Price by {group}')
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, correlation_threshold: float) -> None:
    """
    Plot correlation heatmap of features in DataFrame.
    """

    if correlation_threshold is None:
        plt.figure(figsize=(15, 5))
        sb.heatmap(df.corr(), annot=True, cbar=False)
        plt.show()
        return

    plt.figure(figsize=(15, 5))
    sb.heatmap(df.corr() > correlation_threshold, annot=True, cbar=False)
    plt.show()
