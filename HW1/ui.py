import streamlit as st
import pandas as pd
import numpy as np

from enum import Enum
from operator import add
from scipy.stats import norm
import plotly.graph_objects as go
import math

# ENUM DEFINITIONS


class OptionType(Enum):
    CALL = 1
    PUT = 2


class OrderType(Enum):
    BUY = 1
    SELL = 2

# CLASS DEFINITIONS


class Utils:
    @staticmethod
    def get_option_price(option_type, spot_price, strike_price, risk_free_rate=0.05, volatility=0.25, time=1):
        if volatility == 0:
            volatility = 0.0000001
        if time == 0:
            time = 0.0000001
        d1 = (math.log(spot_price / strike_price) + (risk_free_rate +
              0.5 * volatility**2) * time) / (volatility * math.sqrt(time))
        d2 = d1 - volatility * math.sqrt(time)
        option_price = 0

        if option_type == OptionType.CALL:
            option_price = spot_price * \
                norm.cdf(d1) - strike_price * \
                math.exp(-risk_free_rate * time) * norm.cdf(d2)
        elif option_type == OptionType.PUT:
            option_price = strike_price * \
                math.exp(-risk_free_rate * time) * \
                norm.cdf(-d2) - spot_price * norm.cdf(-d1)

        return option_price

    @staticmethod
    def get_portfolio_details():

        number_col = st.columns(2)

        with number_col[0]:
            num_options = st.number_input(
                "Number of Options:",
                min_value=0, value=1, step=1
            )
        with number_col[1]:
            num_stocks = st.number_input(
                "Number of Stocks:",
                min_value=0, value=0, step=1
            )

        stock_col = st.columns(2)
        with stock_col[0]:
            spot_price = st.number_input(
                "Spot Price:",
                value=100.0, step=1.0
            )

        with stock_col[1]:
            stock_order_type = OrderType.BUY if st.selectbox(
                "Stock Order Type:",
                ["BUY", "SELL"]
            ) == "BUY" else OrderType.SELL

        st.markdown("---")

        checkbox_col = st.columns(3)

        with checkbox_col[0]:
            only_show_cumulative = st.checkbox(
                "Only Show Cumulative", value=False)
        with checkbox_col[1]:
            auto_option_pricing = st.checkbox(
                "Auto Option Pricing", value=True)
        with checkbox_col[2]:
            advanced_options = st.checkbox("Advanced Options", value=False)

        risk_free_rate = 0.05
        volatility = 0.25
        time = 1
        arbitrage_check_type = "Simple"

        if advanced_options:
            st.markdown("---")

            advanced_col = st.columns(4)
            with advanced_col[0]:
                risk_free_rate = st.number_input(
                    "Risk Free Rate:",
                    value=0.05, step=0.01
                )

            with advanced_col[1]:
                volatility = st.number_input(
                    "Volatility:",
                    value=0.25, step=0.01
                )

            with advanced_col[2]:
                time = st.number_input(
                    "Time:",
                    value=1.0, step=0.1
                )

            with advanced_col[3]:
                arbitrage_check_type = st.selectbox(
                    "Arbitrage Check Type:",
                    ["None", "Simple", "Black-Scholes"]
                )

        option_details = []

        st.markdown("---")
        for i in range(num_options):
            st.subheader(f"Option {i+1} Info")

            col1 = st.columns(3)

            with col1[0]:
                option_type = OptionType.CALL if st.selectbox(
                    f"Option {i+1} Type:",
                    ["CALL", "PUT"]
                ) == "CALL" else OptionType.PUT

            with col1[1]:
                option_action = OrderType.BUY if st.selectbox(
                    f"Option {i+1} Action:",
                    ["BUY", "SELL"]
                ) == "BUY" else OrderType.SELL

            with col1[2]:
                strike_price = st.number_input(
                    f"Option {i+1} Strike Price:",
                    min_value=0.1, value=100.0, step=1.0
                )
            option_price = 0
            if not auto_option_pricing:
                option_price = st.number_input(
                    f"Option {i+1} Price:",
                    min_value=0.1, value=0.1, step=1.0
                )
            if not auto_option_pricing and arbitrage_check_type != "None":
                if arbitrage_check_type == "Simple":
                    if option_type == OptionType.CALL:
                        if option_action == OrderType.BUY and spot_price - strike_price > option_price:
                            st.error("Arbitrage Detected!")
                            st.error(
                                f"Stock Price: {spot_price}, Strike Price: {strike_price}, Option Price: {option_price}")
                            st.error(
                                "The difference between the stock price and the strike price is greater than the option price. This is an arbitrage opportunity.")
                        elif option_action == OrderType.SELL and spot_price - strike_price < option_price:
                            st.error("Arbitrage Detected!")
                            st.error(
                                f"Stock Price: {spot_price}, Strike Price: {strike_price}, Option Price: {option_price}")
                            st.error(
                                "The difference between the stock price and the strike price is less than the option price. This is an arbitrage opportunity.")
                    elif option_type == OptionType.PUT:
                        if option_action == OrderType.BUY and strike_price - spot_price > option_price:
                            st.error("Arbitrage Detected!")
                            st.error(
                                f"Stock Price: {spot_price}, Strike Price: {strike_price}, Option Price: {option_price}")
                            st.error(
                                "The difference between the strike price and the stock price is greater than the option price. This is an arbitrage opportunity.")
                        elif option_action == OrderType.SELL and strike_price - spot_price < option_price:
                            st.error("Arbitrage Detected!")
                            st.error(
                                f"Stock Price: {spot_price}, Strike Price: {strike_price}, Option Price: {option_price}")
                            st.error(
                                "The difference between the strike price and the stock price is less than the option price. This is an arbitrage opportunity.")
                elif arbitrage_check_type == "Black-Scholes":
                    black_scholes_price = Utils.get_option_price(
                        option_type, spot_price, strike_price, risk_free_rate, volatility, time)
                    diff = abs(option_price - black_scholes_price)
                    if diff > 0.1:
                        st.error("Arbitrage Detected!")
                        st.error(
                            f"Stock Price: {spot_price}, Strike Price: {strike_price}, Option Price: {option_price}, Black-Scholes Price: ~{int(black_scholes_price * 100) / 100}")
                        st.error(
                            f"The difference between the option price and the Black-Scholes price is ~{int(diff * 100) / 100}. This is an arbitrage opportunity.")

            option_details.append(
                (option_type, option_action, strike_price, option_price)
            )

        return num_options, num_stocks, spot_price, stock_order_type,  option_details, risk_free_rate, volatility, time, auto_option_pricing, arbitrage_check_type, only_show_cumulative

    @staticmethod
    def black_scholes_analysis():
        strike_prices = np.arange(70., 150., 1)
        calls_by_price = [Utils.get_option_price(
            OptionType.CALL, 100, strike_price) for strike_price in strike_prices]
        puts_by_price = [Utils.get_option_price(
            OptionType.PUT, 100, strike_price) for strike_price in strike_prices]

        risk_free_rates = np.arange(0.01, 0.1, 0.01)
        calls_by_strike = [Utils.get_option_price(
            OptionType.CALL, 100, 100, risk_free_rate=risk_free_rate) for risk_free_rate in risk_free_rates]
        puts_by_strike = [Utils.get_option_price(
            OptionType.PUT, 100, 100, risk_free_rate=risk_free_rate) for risk_free_rate in risk_free_rates]

        times = np.arange(0.1, 2, 0.1)
        calls_by_time = [Utils.get_option_price(
            OptionType.CALL, 100, 100, time=time) for time in times]
        puts_by_time = [Utils.get_option_price(
            OptionType.PUT, 100, 100, time=time) for time in times]

        volatilities = np.arange(0.1, 1, 0.1)
        calls_by_volatility = [Utils.get_option_price(
            OptionType.CALL, 100, 100, volatility=volatility) for volatility in volatilities]
        puts_by_volatility = [Utils.get_option_price(
            OptionType.PUT, 100, 100, volatility=volatility) for volatility in volatilities]

        # Create individual plots
        fig_price = go.Figure()
        fig_strike = go.Figure()
        fig_time = go.Figure()
        fig_volatility = go.Figure()

        fig_price.add_trace(go.Scatter(
            x=strike_prices, y=calls_by_price, mode='lines', name='Call by Strike Price'))
        fig_price.add_trace(go.Scatter(
            x=strike_prices, y=puts_by_price, mode='lines', name='Put by Strike Price'))
        fig_price.update_layout(title='Option Prices by Strike Price')

        fig_strike.add_trace(go.Scatter(
            x=risk_free_rates, y=calls_by_strike, mode='lines', name='Call by Risk Free Rate'))
        fig_strike.add_trace(go.Scatter(
            x=risk_free_rates, y=puts_by_strike, mode='lines', name='Put by Risk Free Rate'))
        fig_strike.update_layout(title='Option Prices by Risk Free Rate')

        fig_time.add_trace(go.Scatter(x=times, y=calls_by_time,
                                      mode='lines', name='Call by Time'))
        fig_time.add_trace(go.Scatter(x=times, y=puts_by_time,
                                      mode='lines', name='Put by Time'))
        fig_time.update_layout(title='Option Prices by Time')

        fig_volatility.add_trace(go.Scatter(
            x=volatilities, y=calls_by_volatility, mode='lines', name='Call by Volatility'))
        fig_volatility.add_trace(go.Scatter(
            x=volatilities, y=puts_by_volatility, mode='lines', name='Put by Volatility'))
        fig_volatility.update_layout(title='Option Prices by Volatility')

        # Display the plots
        st.plotly_chart(fig_price, use_container_width=True)
        st.plotly_chart(fig_strike, use_container_width=True)
        st.plotly_chart(fig_time, use_container_width=True)
        st.plotly_chart(fig_volatility, use_container_width=True)

    @staticmethod
    def get_example_portfolios(risk_free_rate, volatility, time):
        option_buy_call_80 = Option(
            OptionType.CALL, OrderType.BUY, 80, 100, False, risk_free_rate, volatility, time)
        option_buy_call_120 = Option(
            OptionType.CALL, OrderType.BUY, 120, 100, False, risk_free_rate, volatility, time)
        option_put_buy_80 = Option(
            OptionType.PUT, OrderType.BUY, 80, 100, False, risk_free_rate, volatility, time)
        option_sell_call_80 = Option(
            OptionType.CALL, OrderType.SELL, 80, 100, False, risk_free_rate, volatility, time)
        option_sell_call_120 = Option(
            OptionType.CALL, OrderType.SELL, 120, 100, False, risk_free_rate, volatility, time)
        option_sell_call_100 = Option(OptionType.CALL, OrderType.SELL, 100, 100,
                                      False, risk_free_rate, volatility, time)
        option_buy_call_120 = Option(
            OptionType.CALL, OrderType.BUY, 120, 100, False, risk_free_rate, volatility, time)
        option_buy_put_120 = Option(
            OptionType.PUT, OrderType.BUY, 120, 100, False, risk_free_rate, volatility, time)

        portfolios = [
            Portfolio("Bull Spreads", [], [
                (option_buy_call_80, 1),
                (option_sell_call_120, 1)]),

            Portfolio("Bear Spreads", [], [
                (option_sell_call_80, 1),
                (option_buy_call_120, 1)]),

            Portfolio("Butterfly Spreads", [], [
                (option_buy_call_80, 1),
                (option_sell_call_100, 2),
                (option_buy_call_120, 1)]),

            Portfolio("Straddles", [], [
                (option_buy_call_80, 1),
                (option_put_buy_80, 1)]),

            Portfolio("Straps", [], [
                (option_buy_call_80, 3),
                (option_put_buy_80, 1)]),

            Portfolio("Strips", [], [
                (option_buy_call_80, 1),
                (option_put_buy_80, 3)]),

            Portfolio("Strangle", [], [
                (option_buy_call_80, 1),
                (option_buy_put_120, 1)]),

            Portfolio("Covered Call", [
                (Stock(100, OrderType.BUY), 1)],
                [(option_sell_call_80, 1)])
        ]

        return portfolios


class Option:
    def __init__(self, option_type, order_type, strike_price, spot_price, option_price=False, risk_free_rate=0.05, volatility=0.25, time=1):
        self.option_type = option_type
        self.order_type = order_type
        self.strike_price = strike_price
        self.spot_price = spot_price
        self.option_price = option_price if option_price else Utils.get_option_price(
            option_type, spot_price, strike_price, risk_free_rate, volatility, time)

    def value(self, point):
        result = 0
        difference = point - self.strike_price

        if self.option_type == OptionType.CALL:
            if self.order_type == OrderType.BUY:
                if point < self.strike_price:
                    result = -self.option_price
                else:
                    result = difference - self.option_price
            elif self.order_type == OrderType.SELL:
                if point < self.strike_price:
                    result = self.option_price
                else:
                    result = self.option_price - difference
        elif self.option_type == OptionType.PUT:
            if self.order_type == OrderType.BUY:
                if point > self.strike_price:
                    result = -self.option_price
                else:
                    result = -difference - self.option_price
            elif self.order_type == OrderType.SELL:
                if point > self.strike_price:
                    result = self.option_price
                else:
                    result = difference + self.option_price

        return result

    def __str__(self) -> str:

        order_type_str = "BUY" if self.order_type == OrderType.BUY else "SELL"
        option_type_str = "CALL" if self.option_type == OptionType.CALL else "PUT"

        return f"OPTION ({order_type_str} {option_type_str} {self.strike_price})"


class Stock:
    def __init__(self, price, order_type: OrderType):
        self.price = price
        self.order_type = order_type

    def value(self, point):
        difference = point - self.price

        if self.order_type == OrderType.BUY:
            return difference
        elif self.order_type == OrderType.SELL:
            return -difference

        return 0

    def __str__(self) -> str:
        order_type_str = "BUY" if self.order_type == OrderType.BUY else "SELL"
        return f"STOCK ({order_type_str})"


class Portfolio:
    def __init__(self, name, stocks, options):
        self.name = name
        self.stocks = stocks
        self.options = options

    def visualize(self, step, separate=False):
        end = 0
        for (stock, quantity) in self.stocks:
            end = max(end, stock.price)
        for (option, quantity) in self.options:
            end = max(end, option.strike_price)
            end = max(end, option.spot_price)
        x = np.arange(0, end + 50, step)
        y = np.zeros(len(x))
        strike_prices = [option.strike_price for (
            option, quantity) in self.options]

        max_y = -math.inf
        min_y = math.inf
        for (stock, quantity) in self.stocks:
            _y = [stock.value(point) * quantity for point in x]
            max_y = max(max_y, max(_y))
            min_y = min(min_y, min(_y))

        for (option, quantity) in self.options:
            _y = [option.value(point) * quantity for point in x]
            max_y = max(max_y, max(_y))
            min_y = min(min_y, min(_y))

        fig = go.Figure()

        # x axis line
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=end + 50,
            y1=0,
            line=dict(color="gray", width=2)
        )

        # y axis line
        fig.add_shape(
            type="line",
            x0=0,
            y0=min_y,
            x1=0,
            y1=max_y,
            line=dict(color="gray", width=2)
        )

        for (stock, quantity) in self.stocks:
            _y = [stock.value(point) * quantity for point in x]
            y = list(map(add, y, _y))

            if separate:
                fig.add_trace(go.Scatter(
                    x=x, y=_y, mode='lines', name=str(stock if quantity == 1 else f"{quantity}x {stock}")))

        for (option, quantity) in self.options:
            _y = [option.value(point) * quantity for point in x]
            y = list(map(add, y, _y))

            if separate:
                fig.add_trace(go.Scatter(
                    x=x, y=_y, mode='lines', name=(str(option if quantity == 1 else f"{quantity}x {option}"))))

        # if not separate:
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                      name='Combined Portfolio Value'))

        fig.add_trace(go.Scatter(x=strike_prices, y=np.zeros(len(
            strike_prices)), mode='markers', marker=dict(color='red'), name='Strike Prices'))

        fig.update_layout(
            showlegend=True,
            xaxis=dict(title='Stock Price'),
            yaxis=dict(title='Profit/Loss'),
            title=self.name,
        )

        st.plotly_chart(fig)
        st.markdown("---")

    def add_stock(self, stock, quantity=1):
        s = Stock(stock.price, stock.order_type)
        self.stocks.append((s, quantity))

    def add_option(self, option, quantity):
        o = Option(option.option_type, option.order_type,
                   option.strike_price, option.spot_price, option.option_price)
        self.options.append((o, quantity))


pages = {
    "Home": "home",
    "Create Portfolio": "create_portfolio",
    "Example Portfolios": "example_portfolios",
    "Black Scholes Model": "black_scholes"
}

st.sidebar.title("Emir Soyturk")
selection = st.sidebar.radio("Go to", list(pages.keys()))

if selection == "Example Portfolios":
    st.title("Example Portfolios")
    st.markdown("""
        Example portfolios showcase different option trading strategies that you can explore and analyze. These portfolios are constructed using a combination of options and stocks, and their payoff profiles vary based on market conditions and strategy objectives.
        
        Each example portfolio consists of a name, a combination of stocks, and a set of options. By visualizing the portfolio's payoff, you can gain insights into how the portfolio performs under different scenarios and market conditions.
        
        You can adjust the risk-free rate, volatility, and time to customize the example portfolios based on your preferences and analyze their potential outcomes.
        
        Explore the example portfolios to learn about various option trading strategies and their potential payoff profiles.
    """)

    col = st.columns(3)

    with col[0]:
        risk_free_rate = st.number_input(
            "Risk Free Rate:", value=0.05, step=0.01)
    with col[1]:
        volatility = st.number_input("Volatility:", value=0.25, step=0.01)
    with col[2]:
        time = st.number_input("Time:", value=1.0, step=0.1)
    only_show_cumulative = st.checkbox(
        "Only Show Cumulative", value=False)

    portfolios = Utils.get_example_portfolios(risk_free_rate, volatility, time)

    for portfolio in portfolios:
        portfolio.visualize(1, not only_show_cumulative)

elif selection == "Create Portfolio":
    st.title("Create a New Portfolio")
    st.markdown("""
        Create your own portfolio by selecting the number of stocks and options, providing their details, and visualizing the payoff profile of the portfolio.
        
        - Select the number of stocks and options, and provide their details.
        - Visualize the payoff profile of the portfolio.
    """)

    portfolio = Portfolio("Portfolio", [], [])
    num_options, num_stocks, stock_price, stock_order_type, option_details, risk_free_rate, volatility, time, auto_option_pricing, arbitrage_check_type, only_show_cumulative = Utils.get_portfolio_details()

    if not auto_option_pricing:
        if arbitrage_check_type == "Simple":
            for i in range(num_options):
                option_type_1, option_action_1, strike_price_1, option_price_1 = option_details[
                    i]
                for j in range(i+1, num_options):
                    option_type_2, option_action_2, strike_price_2, option_price_2 = option_details[
                        j]

                    # Check for arbitrage opportunity between pairs of options
                    if option_type_1 == option_type_2 and option_action_1 == option_action_2:
                        if option_type_1 == OptionType.CALL:
                            if option_action_1 == OrderType.BUY:
                                if strike_price_2 > strike_price_1 and option_price_2 > option_price_1:
                                    st.error("Arbitrage Detected!")
                                    st.error(
                                        f"Strike Price {i + 1}: {strike_price_1}, Strike Price {j + 1}: {strike_price_2}, Option Price {i + 1}: {option_price_1}, Option Price {j + 1}: {option_price_2}")
                                    st.error(
                                        f"The strike price of the {j + 1}th option is greater than the strike price of the {i + 1}th option, and the option price of the {j + 1}th option is greater than the option price of the {i + 1}th option. This is an arbitrage opportunity.")
                            elif option_action_1 == OrderType.SELL:
                                if strike_price_1 > strike_price_2 and option_price_1 > option_price_2:
                                    st.error("Arbitrage Detected!")
                                    st.error(
                                        f"Strike Price {i + 1}: {strike_price_1}, Strike Price {j + 1}: {strike_price_2}, Option Price {i + 1}: {option_price_1}, Option Price {j + 1}: {option_price_2}")
                                    st.error(
                                        f"The strike price of the {i + 1}th option is greater than the strike price of the {j + 1}th option, and the option price of the {i + 1}th option is greater than the option price of the {j + 1}th option. This is an arbitrage opportunity.")
                        elif option_type_1 == OptionType.PUT:
                            if option_action_1 == OrderType.BUY:
                                if strike_price_1 > strike_price_2 and option_price_1 > option_price_2:
                                    st.error("Arbitrage Detected!")
                                    st.error(
                                        f"Strike Price {i + 1}: {strike_price_1}, Strike Price {j + 1}: {strike_price_2}, Option Price {i + 1}: {option_price_1}, Option Price {j + 1}: {option_price_2}")
                                    st.error(
                                        f"The strike price of the {i + 1}th option is greater than the strike price of the {j + 1}th option, and the option price of the {i + 1}th option is greater than the option price of the {j + 1}th option. This is an arbitrage opportunity.")
                            elif option_action_1 == OrderType.SELL:
                                if strike_price_2 > strike_price_1 and option_price_2 > option_price_1:
                                    st.error("Arbitrage Detected!")
                                    st.error(
                                        f"Strike Price {i + 1}: {strike_price_1}, Strike Price {j + 1}: {strike_price_2}, Option Price {i + 1}: {option_price_1}, Option Price {j + 1}: {option_price_2}")
                                    st.error(
                                        f"The strike price of the {j + 1}th option is greater than the strike price of the {i + 1}th option, and the option price of the {j + 1}th option is greater than the option price of the {i + 1}th option. This is an arbitrage opportunity.")

    if num_stocks > 0:
        portfolio.add_stock(Stock(stock_price, stock_order_type), num_stocks)

    for i in range(num_options):
        option_type, option_action, strike_price, option_price = option_details[i]
        option_price = option_price if not auto_option_pricing else False
        option = Option(option_type, option_action, strike_price,
                        stock_price, option_price, risk_free_rate, volatility, time)
        portfolio.add_option(option, 1)

    portfolio.visualize(1, not only_show_cumulative)

elif selection == "Black Scholes Model":
    st.title("Black Scholes Model")
    st.markdown("""
        The Black-Scholes model is a mathematical model used to calculate the theoretical price of options. It provides a framework for pricing European-style options, which can only be exercised at expiration. The model was developed by economists Fischer Black and Myron Scholes in 1973.
        
        The Black-Scholes model calculates the fair value of an option by considering factors such as the current price of the underlying asset, the strike price of the option, the time to expiration, the risk-free interest rate, and the volatility of the underlying asset.
        
        The model provides insights into the factors that influence option prices, such as changes in the underlying asset's price, time to expiration, and volatility. It has been widely used in options trading and has contributed to the development of various option pricing and trading strategies.
    """)

    st.markdown("---")
    Utils.black_scholes_analysis()

elif selection == "Home":
    st.title("Welcome to Option Visualizer")
    st.markdown("""
        This is a simple tool to visualize the payoff of your option strategies. Options are financial derivatives that give the holder the right, but not the obligation, to buy or sell an underlying asset at a predetermined price within a specified time period.
        
        The main purposes of options include:
        - **Hedging**: Protecting against potential losses in an existing investment.
        - **Speculation**: Profiting from market movements without owning the underlying asset.
        - **Income Generation**: Selling options to collect premiums and generate income.
        - **Risk Management**: Using options to manage and control risk exposure.
        - **Leverage**: Controlling larger positions with a smaller investment.
        
        Please keep in mind that options trading involves risks and complexities. It's important to have a good understanding of the underlying assets, market conditions, and option strategies before engaging in options trading.
    """)

    st.markdown("---")
