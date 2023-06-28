import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")


class CryptoPredictor:
    def __init__(self, file_path):
        self.df = self.load_data(file_path)
        self.scaler = MinMaxScaler()

    @staticmethod
    def load_data(file_path):
        df = pd.read_csv(file_path)
        df.columns = ['open_time', 'open_price', 'high_price',
                      'low_price', 'close_price', 'volume', 'close_time']
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df['open_time'] = pd.to_datetime(df['open_time'] * 1000, unit='ms')
        return df

    @staticmethod
    def plot_close_price_history(df):
        plt.figure(figsize=(12.5, 4.5))
        plt.plot(df['close_time'], df['close_price'], label='ETH Close Price')
        plt.title('ETH Close Price History')
        plt.xlabel('Date')
        plt.ylabel('Price USD ($)')
        plt.legend(loc='upper left')
        plt.show()

    @staticmethod
    def slice_data_by_date(df, start_date):
        return df[df['close_time'] >= start_date].copy()

    @staticmethod
    def add_moving_averages(sliced_df, ma_days):
        for ma in ma_days:
            column_name = f"MA for {ma} days"
            sliced_df = sliced_df.assign(
                **{column_name: sliced_df['close_price'].rolling(ma).mean()})
        return sliced_df

    @staticmethod
    def plot_moving_averages(sliced_df, ma_days):
        plt.figure(figsize=(12.5, 4.5))
        plt.plot(sliced_df['close_time'],
                 sliced_df['close_price'], label='ETH Close Price')
        for ma in ma_days:
            column_name = f"MA for {ma} days"
            plt.plot(sliced_df['close_time'],
                     sliced_df[column_name], label=column_name)
        plt.title('ETH Close Price History')
        plt.xlabel('Date')
        plt.ylabel('Price USD ($)')
        plt.legend(loc='upper left')
        plt.show()

    @staticmethod
    def add_daily_returns(sliced_df):
        return sliced_df.assign(daily_return=sliced_df['close_price'].pct_change())

    # Other functions remain similar to what you had earlier, with minor changes to avoid SettingWithCopyWarning

    @staticmethod
    def create_lstm_model(input_shape):
        model = Sequential()
        model.add(LSTM(500, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_lstm_model(self, df, test_size=0.2):
        data = df.filter(['close_price'])
        dataset = data.values
        training_data_len = int(np.ceil(len(dataset) * (1 - test_size)))

        scaled_data = self.scaler.fit_transform(dataset)

        train_data = scaled_data[0:int(training_data_len), :]
        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = self.create_lstm_model((x_train.shape[1], 1))
        model.fit(x_train, y_train, batch_size=1, epochs=10)

        return model, training_data_len

    def evaluate_model(self, model, df, training_data_len):
        data = df.filter(['close_price'])
        dataset = data.values
        scaled_data = self.scaler.transform(dataset)

        test_data = scaled_data[training_data_len - 60:, :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions = model.predict(x_test)
        predictions = self.scaler.inverse_transform(predictions)

        rmse = np.sqrt(mean_squared_error(predictions, y_test))

        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions

        plt.figure(figsize=(16, 8))
        plt.title('Model')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.plot(train['close_price'])
        plt.plot(valid[['close_price', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()

        return rmse


crypto_predictor = CryptoPredictor('../data/1day.csv')
sliced_df = crypto_predictor.slice_data_by_date(
    crypto_predictor.df, '2023-01-01')
model, training_data_len = crypto_predictor.train_lstm_model(sliced_df)
rmse = crypto_predictor.evaluate_model(model, sliced_df, training_data_len)

print(f'Root Mean Squared Error: {rmse}')
