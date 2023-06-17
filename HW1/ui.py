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
        num_options = st.number_input(
            "Number of Options:",
            min_value=0, value=1, step=1
        )
        num_stocks = st.number_input(
            "Number of Stocks:",
            min_value=0, value=0, step=1
        )
        spot_price = st.number_input(
            "Spot Price:",
            value=100.0, step=1.0
        )

        risk_free_rate = st.number_input(
            "Risk Free Rate:",
            value=0.05, step=0.01
        )
        volatility = st.number_input(
            "Volatility:",
            value=0.25, step=0.01
        )
        time = st.number_input(
            "Time:",
            value=1.0, step=0.1
        )

        auto_option_pricing = st.checkbox("Auto Option Pricing", value=True)

        option_details = []

        for i in range(num_options):
            st.subheader(f"Option {i+1} Info")
            option_type = st.selectbox(
                f"Option {i+1} Type:",
                ["CALL", "PUT"]
            )
            option_action = st.selectbox(
                f"Option {i+1} Action:",
                ["BUY", "SELL"]
            )
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
            option_details.append(
                (option_type, option_action, strike_price, option_price)
            )

        return num_options, num_stocks, spot_price, option_details, risk_free_rate, volatility, time, auto_option_pricing

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
        portfolios = [
            Portfolio("Bull Bear Spreads", [], [
                (Option(OptionType.CALL, OrderType.BUY, 80, 100,
                        False, risk_free_rate, volatility, time), 1),
                (Option(OptionType.CALL, OrderType.SELL, 120, 100, False, risk_free_rate, volatility, time), 1)]),

            Portfolio("Bear Spreads", [], [
                (Option(OptionType.CALL, OrderType.SELL, 80, 100,
                        False, risk_free_rate, volatility, time), 1),
                (Option(OptionType.CALL, OrderType.BUY, 120, 100, False, risk_free_rate, volatility, time), 1)]),

            Portfolio("Butterfly Spreads", [], [
                (Option(OptionType.CALL, OrderType.BUY, 80, 100,
                        False, risk_free_rate, volatility, time), 1),
                (Option(OptionType.CALL, OrderType.SELL, 100, 100,
                        False, risk_free_rate, volatility, time), 2),
                (Option(OptionType.CALL, OrderType.BUY, 120, 100, False, risk_free_rate, volatility, time), 1)]),

            Portfolio("Straddles", [], [
                (Option(OptionType.CALL, OrderType.BUY, 80, 100,
                        False, risk_free_rate, volatility, time), 1),
                (Option(OptionType.PUT, OrderType.BUY, 80, 100, False, risk_free_rate, volatility, time), 1)]),

            Portfolio("Straps", [], [
                (Option(OptionType.CALL, OrderType.BUY, 80, 100,
                        False, risk_free_rate, volatility, time), 3),
                (Option(OptionType.PUT, OrderType.BUY, 80, 100, False, risk_free_rate, volatility, time), 1)]),

            Portfolio("Strips", [], [
                (Option(OptionType.CALL, OrderType.BUY, 80, 100,
                        False, risk_free_rate, volatility, time), 1),
                (Option(OptionType.PUT, OrderType.BUY, 80, 100, False, risk_free_rate, volatility, time), 3)]),

            Portfolio("Strangle", [], [
                (Option(OptionType.CALL, OrderType.BUY, 80, 100,
                        False, risk_free_rate, volatility, time), 1),
                (Option(OptionType.PUT, OrderType.BUY, 120, 100, False, risk_free_rate, volatility, time), 1)]),

            Portfolio("Covered Call", [
                (Stock(100), 1)],
                [(Option(OptionType.CALL, OrderType.SELL, 80, 100, False, risk_free_rate, volatility, time), 1)])
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
        name = ""
        if self.order_type == OrderType.BUY:
            name += "BUY"
        elif self.order_type == OrderType.SELL:
            name += "SELL"
        name += " "
        if self.option_type == OptionType.CALL:
            name += "CALL"
        elif self.option_type == OptionType.PUT:
            name += "PUT"

        return name


class Stock:
    def __init__(self, price):
        self.price = price

    def value(self, point):
        return point

    def __str__(self) -> str:
        return "STOCK"


class Portfolio:
    def __init__(self, name, stocks, options):
        self.name = name
        self.stocks = stocks
        self.options = options

    def visualize(self, start, end, step, separate=False):
        x = np.arange(start, end, step)
        y = np.zeros(len(x))
        strike_prices = [option.strike_price for (
            option, quantity) in self.options]

        fig = go.Figure()

        for (stock, quantity) in self.stocks:
            _y = [stock.value(point) * quantity for point in x]
            y = list(map(add, y, _y))

            if separate:
                fig.add_trace(go.Scatter(
                    x=x, y=_y, mode='lines', name=str(stock)))

        for (option, quantity) in self.options:
            _y = [option.value(point) * quantity for point in x]
            y = list(map(add, y, _y))

            if separate:
                fig.add_trace(go.Scatter(
                    x=x, y=_y, mode='lines', name=str(option)))

        # if not separate:
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                      name='Combined Portfolio Value'))

        fig.add_trace(go.Scatter(x=strike_prices, y=np.zeros(len(
            strike_prices)), mode='markers', marker=dict(color='red'), name='Strike Prices'))

        fig.update_layout(
            showlegend=True,
            xaxis=dict(title='Stock Price'),
            yaxis=dict(title='Portfolio Value'),
            title=self.name,
            width=600,
            height=400
        )

        st.plotly_chart(fig)
        st.markdown("---")

    def add_stock(self, stock, quantity):
        s = Stock(stock.price)
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
    st.markdown("Example portfolios showcase different option trading strategies that you can explore and analyze. These portfolios are constructed using a combination of options and stocks, and their payoff profiles vary based on market conditions and strategy objectives.")

    st.markdown("Each example portfolio consists of a name, a combination of stocks, and a set of options. By visualizing the portfolio's payoff, you can gain insights into how the portfolio performs under different scenarios and market conditions.")

    st.markdown("You can adjust the risk-free rate, volatility, and time to customize the example portfolios based on your preferences and analyze their potential outcomes.")

    st.markdown(
        "Explore the example portfolios to learn about various option trading strategies and their potential payoff profiles.")

    risk_free_rate = st.number_input(
        "Risk Free Rate:",
        value=0.05, step=0.01
    )
    volatility = st.number_input(
        "Volatility:",
        value=0.25, step=0.01
    )
    time = st.number_input(
        "Time:",
        value=1.0, step=0.1
    )

    portfolios = Utils.get_example_portfolios(risk_free_rate, volatility, time)

    for portfolio in portfolios:
        portfolio.visualize(-100, 200, 5, True)

elif selection == "Create Portfolio":
    st.title("Create a New Portfolio")
    st.markdown("Create your own portfolio by selecting the number of stocks and options, providing their details, and visualizing the payoff profile of the portfolio.")

    portfolio = Portfolio("Portfolio", [], [])
    num_options, num_stocks, stock_price, option_details, risk_free_rate, volatility, time, auto_option_pricing = Utils.get_portfolio_details()
    portfolio.add_stock(Stock(stock_price), num_stocks)

    for i in range(num_options):
        option_type, option_action, strike_price, option_price = option_details[i]
        option_type = OptionType.CALL if option_type == "CALL" else OptionType.PUT
        order_type = OrderType.BUY if option_action == "BUY" else OrderType.SELL
        option_price = option_price if not auto_option_pricing else False
        option = Option(
            option_type, order_type, strike_price, stock_price, option_price, risk_free_rate, volatility, time
        )
        portfolio.add_option(option, 1)

    portfolio.visualize(-100, 200, 5, True)

elif selection == "Black Scholes Model":
    st.title("Black Scholes Model")
    st.markdown("The Black-Scholes model is a mathematical model used to calculate the theoretical price of options. It provides a framework for pricing European-style options, which can only be exercised at expiration. The model was developed by economists Fischer Black and Myron Scholes in 1973, and it assumes several key factors:")
    st.markdown(
        "- The price of the underlying asset follows a geometric Brownian motion.")
    st.markdown(
        "- The market is efficient and there are no transaction costs or restrictions on short-selling.")
    st.markdown("- The risk-free interest rate is known and constant.")
    st.markdown(
        "- The volatility of the underlying asset's returns is constant.")

    st.markdown("The Black-Scholes model calculates the fair value of an option by considering these factors and using mathematical formulas. It takes into account the current price of the underlying asset, the strike price of the option, the time to expiration, the risk-free interest rate, and the volatility of the underlying asset.")

    st.markdown("The model provides insights into the factors that influence option prices, such as changes in the underlying asset's price, time to expiration, and volatility. It has been widely used in options trading and has contributed to the development of various option pricing and trading strategies.")

    st.markdown("---")
    Utils.black_scholes_analysis()

elif selection == "Home":
    st.title("Welcome to Option Visualizer")
    st.markdown("This is a simple tool to visualize the payoff of your option strategies. Options are financial derivatives that give the holder the right, but not the obligation, to buy or sell an underlying asset at a predetermined price within a specified time period.")

    st.markdown("The main purposes of options include:")
    st.markdown(
        "- **Hedging**: Protecting against potential losses in an existing investment.")
    st.markdown(
        "- **Speculation**: Profiting from market movements without owning the underlying asset.")
    st.markdown(
        "- **Income Generation**: Selling options to collect premiums and generate income.")
    st.markdown(
        "- **Risk Management**: Using options to manage and control risk exposure.")
    st.markdown(
        "- **Leverage**: Controlling larger positions with a smaller investment.")

    st.markdown("Please keep in mind that options trading involves risks and complexities. It's important to have a good understanding of the underlying assets, market conditions, and option strategies before engaging in options trading.")

    st.markdown("---")
