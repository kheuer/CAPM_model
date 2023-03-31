import hashlib
import yaml
import requests
from bs4 import BeautifulSoup
from api import *
import numpy as np
import pynput.mouse
import yaml
import time
import datetime
import logging
import mysql.connector
from pynput.mouse import Listener
import matplotlib.pyplot as plt
import scipy.optimize


fmp = FinancialModelingPrep()

with open("credentials.yaml", "r") as file:
    credentials = yaml.load(file, Loader=yaml.FullLoader)

with open("settings.yaml", "r") as file:
    settings_container = yaml.load(file, Loader=yaml.FullLoader)

all_ticker_symbols, _ = fmp.call_all_available_tickers()



prices = {}
def get_closing_prices(ticker_symbol):
    try:
        return prices[ticker_symbol]
    except KeyError:
        timeseries, meta = fmp.call_close_timeseries(ticker_symbol)
        prices[ticker_symbol] = [dp["adjClose"] for dp in reversed(timeseries)]
        return get_closing_prices(ticker_symbol)

company_names = {}
def get_company_name(ticker_symbol):
    try:
        return company_names[ticker_symbol]
    except KeyError:
        info, _ = fmp.call_profile(ticker_symbol)
        company_names[ticker_symbol] = info["companyName"]
        return get_company_name(ticker_symbol)

class PortfolioManager:
    def __init__(self):
        self._portfolios = {}

    def add(self, assets, min_position_weight, max_position_weight):
        key = str(assets) + str(min_position_weight) + str(max_position_weight)
        if key in self._portfolios:
            print("portfolio already saved, skipping")
            return
        else:
            data = []
            desc = []
            colors = []
            weights_container = {}
            risk_free_container = {}

            for ticker_symbol in assets:
                if ticker_symbol not in prices:
                    i = 0
                    while True:
                        try:
                            price, meta = fmp.call_price(ticker_symbol, "USD")
                            break
                        except ApiRateError:
                            i += 1
                        if i == 5:
                            raise ApiRateError("Could not call price, gave up after 5 attempts")

            portfolio = Portfolio(assets)
            msr_weights = portfolio.get_msr_weights(min_position_weight=min_position_weight,
                                                    max_position_weight=max_position_weight)

            # get a few efficiency-frontier portfolios
            efficiency_frontier = {}
            for i in range(100 * len(portfolio)):
                random_weights = portfolio.get_random_weights(min_position_weight=min_position_weight,
                                                              max_position_weight=max_position_weight)
                sd = portfolio.get_standard_deviation(random_weights)
                er = portfolio.get_expected_return(random_weights)
                add = True
                for (sd_efficient, er_efficient) in efficiency_frontier:
                    if sd <= sd_efficient and er >= er_efficient:
                        efficiency_frontier.pop((sd_efficient, er_efficient,))
                        break
                    elif sd >= sd_efficient and er <= er_efficient:
                        add = False
                        break
                if add:
                    efficiency_frontier[(sd, er,)] = random_weights

            # add capital market line portfolios
            for percent_risk_free in np.linspace(-0.5, 0.25, 51):
                percent_in_portfolio = 1 - percent_risk_free
                cml_weights = portfolio.get_cml_weights(msr_weights, percent_risk_free)
                query_weights = np.array(cml_weights) / (sum(cml_weights))
                sd = percent_in_portfolio * portfolio.get_standard_deviation(query_weights)
                er = percent_in_portfolio * portfolio.get_expected_return(query_weights) + percent_risk_free * portfolio.risk_free_rate
                data.append({"x": sd, "y": er})
                desc.append({"type": "portfolio",
                             "desc": "Capital Market Line Portfolio"})
                colors.append("#e6fa4d")  # Yellow
                weights_container[(sd, er)] = cml_weights
                risk_free_container[(sd, er)] = percent_risk_free

            # add the efficiency frontier
            for (sd, er), weights in efficiency_frontier.items():
                data.append({"x": sd, "y": er})
                desc.append({"type": "asset",
                             "desc": "Efficiency Frontier Portfolio"})
                colors.append("#cccccc")  # Grey
                weights_container[(sd, er)] = weights
                risk_free_container[(sd, er)] = 0

            # plot single assets
            for i, pos in enumerate(portfolio.positions):
                sd = pos.standard_deviation * np.sqrt(252)
                er = (1 + pos.expected_return) ** 252 - 1
                data.append({"x": sd, "y": er})
                desc.append({"type": "asset",
                             "desc": get_company_name(pos.ticker_symbol)
                             })
                colors.append("#c33d21")  # Red
                weights = [0 for x in range(len(portfolio))]
                weights[i] = 1
                weights_container[(sd, er)] = list(weights)
                risk_free_container[(sd, er)] = 0

            # add mvp
            mvp_weights = portfolio.get_mvp_weights(min_position_weight=min_position_weight,
                                                    max_position_weight=max_position_weight)
            sd = portfolio.get_standard_deviation(mvp_weights)
            er = portfolio.get_expected_return(mvp_weights)

            data.append(portfolio._get_scatter_data(mvp_weights))
            desc.append({"type": "portfolio",
                         "desc": "Minimum Variance Portfolio"})
            colors.append("#213fc3")    # Blue
            weights_container[(sd, er,)] = list(mvp_weights)
            risk_free_container[(sd, er,)] = 0


            # add maximum sharpe ratio portfolio
            sd = portfolio.get_standard_deviation(msr_weights)
            er = portfolio.get_expected_return(msr_weights)

            data.append(portfolio._get_scatter_data(msr_weights))
            desc.append({"type": "portfolio",
                         "desc": "Maximum Sharpe Ratio Portfolio"})
            colors.append("#213fc3")  # Blue
            weights_container[(sd, er,)] = list(msr_weights)
            risk_free_container[(sd, er)] = 0

            self._portfolios[key] = {
                "instance": portfolio,
                "data": data,
                "description": desc,
                "colors": colors,
                "weights": weights_container,
                "percentages_risk_free": risk_free_container,
                "lookup_array": [str(d["x"])+str(d["y"]) for d in data],
                "company_names": [get_company_name(x.ticker_symbol) for x in portfolio.positions],
                "ticker_symbols": [x.ticker_symbol for x in portfolio.positions]
            }

    def get_data(self, assets, min_position_weight, max_position_weight):
        key = str(assets) + str(min_position_weight) + str(max_position_weight)
        if key not in self._portfolios:
            print("adding data")
            self.add(assets, min_position_weight, max_position_weight)
        return self._portfolios[key]

def check_credentials(username, password):
    password += "salt"  # this should really be a random string
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    try:
        if password_hash == credentials[username]:
            return True
        else:
            return False
    except KeyError:
        return False


def get_asset_descriptor(ticker_symbol):
    prices = [round(dp, 2) for dp in get_closing_prices(ticker_symbol)]
    change = round((prices[-1] - prices[-7]) / prices[-7]*100, 2)

    return {"ticker_symbol": ticker_symbol,
            "company_name": get_company_name(ticker_symbol),
            "price": prices[-1],
            "change": change,   # describes last weeks change
            }


def get_daily_returns(ticker_symbol):
    closing_prices = get_closing_prices(ticker_symbol)
    daily_returns = []
    for closing_price_index in range(len(closing_prices) - 1):
        later_price = closing_prices[closing_price_index + 1]
        current_price = closing_prices[closing_price_index]
        change = round(float((later_price - current_price) / current_price), 6)
        daily_returns.append(change)
    return daily_returns


class Portfolio:
    # portfolio sd and er are yearly for easier interpretation
    def __init__(self, positions, risk_free_rate=0.03):
        self.risk_free_rate = risk_free_rate
        data_errors = []
        self.positions = []
        for i, ticker_symbol in enumerate(positions):
            try:
                asset = Asset(ticker_symbol, self)
                self.positions.append(asset)
            except DataUnavailableError:
                data_errors.append(ticker_symbol)
        if data_errors:
            logger.warning(f"data is unavailable for {data_errors}, excluding from portfolio")
        print(
            f"initialized portfolio with ticker_symbols {[asset.ticker_symbol for asset in self.positions]}.")

        stock_prices = []
        for i, position in enumerate(self.positions):
            try:
                price = prices[position.ticker_symbol]
            except KeyError:
                price, meta = fmp.call_price(position.ticker_symbol, "USD")

            stock_prices.append(price)
        self.stock_prices = np.array(stock_prices)

        self._listener = Listener(on_click=self.on_click)
        self._mouse_pressed = False
        self._mouse_checked_time = 0

    def get_covariance(self, i, j):
        len_i = len(self[i].daily_returns)
        len_j = len(self[j].daily_returns)
        min_len = min(len_i, len_j)

        # shorten both arrays to the same length
        i_daily_returns = self[i].daily_returns[len_i - min_len:]
        j_daily_returns = self[j].daily_returns[len_j - min_len:]
        assert len(i_daily_returns) == len(j_daily_returns)

        return np.cov(i_daily_returns, j_daily_returns)[0, 1]

    def get_expected_return(self, weights):
        assert 0.999999 <= sum(weights) <= 1.00001
        expected_return = 0
        for i, weight in enumerate(weights):
            expected_return += weight * self[i].expected_return
        return (expected_return + 1) ** 252 - 1  # annualize daily returns (252 trading days in a year)

    def get_variance(self, weights):
        assert 0.999999 <= sum(weights) <= 1.00001
        variance = 0
        for i, weight_i in enumerate(weights):
            for j, weight_j in enumerate(weights):
                variance += weight_i * weight_j * self.get_covariance(i, j)
        return ((variance ** 0.5) * np.sqrt(252)) ** 2      # annualize daily variance (252 trading days in a year)

    def get_standard_deviation(self, weights):
        
        return self.get_variance(weights) ** 0.5

    def get_mvp_weights(self, min_position_weight=0.00, max_position_weight=1):
        initial_weights = np.array([1 / len(self)] * len(self))
        if max_position_weight * len(self) < 0.9999999999:
            raise AttributeError(
                f"max_position_weight {max_position_weight} and n_positions {len(self.positions)} not compatible.")
        bounds = [[min_position_weight, max_position_weight]] * len(self)
        optim = scipy.optimize.minimize(self.get_variance, initial_weights, bounds=bounds, constraints=
        {"type": "eq", "fun": lambda x: 1 - sum(x)}, tol=0.0000000001)
        if not optim["success"]:
            raise RuntimeError(f"Failed to optimize, reason: {optim['message']}")
        optimal_weights = optim["x"]
        if not 0.9999999999 <= sum(optimal_weights) <= 1.0000000001:
            raise RuntimeError(
                f"Warning: optimal_weights do not sum up to 1 (sum={sum(optimal_weights)}), return values {optim}")
        return optimal_weights

    def get_msr_weights(self, min_position_weight=0.00, max_position_weight=1):
        initial_weights = np.array([1 / len(self)] * len(self))
        if max_position_weight * len(self) < 0.9999999999:
            raise AttributeError(
                f"max_position_weight {max_position_weight} and n_positions {len(self.positions)} not compatible.")
        bounds = [[min_position_weight, max_position_weight]] * len(self)
        objective_fn = lambda weights: 0 - self.get_sharpe_ratio(weights)
        optim = scipy.optimize.minimize(objective_fn, initial_weights, bounds=bounds, constraints=
        {"type": "eq", "fun": lambda x: 1 - sum(x)}, tol=0.0000000001)
        if not optim["success"]:
            raise RuntimeError(f"Failed to optimize, reason: {optim['message']}")
        optimal_weights = optim["x"]
        if not 0.9999999999 <= sum(optimal_weights) <= 1.0000000001:
            raise RuntimeError(
                f"Warning: optimal_weights do not sum up to 1 (sum={sum(optimal_weights)}), return values {optim}")
        return optimal_weights

    def get_utility(self, weights, risk_aversion=1):
        return self.get_expected_return(weights) - 0.5 * risk_aversion * self.get_variance(weights)

    def get_sharpe_ratio(self, weights):
        er = self.get_expected_return(weights)
        sd = self.get_standard_deviation(weights)
        return (er - self.risk_free_rate) / sd

    def get_cml_weights(self, msr_weights, percent_risk_free):
        """

        :param msr_weights: weights of the maximum sharp ratio portfolio
        :param percent_risk_free: describes what percentage of the portfolio is invested at risk free rate
        :return:
        """
        percent_in_portfolio = 1 - percent_risk_free

        cml_portfolio = np.array(msr_weights) * percent_in_portfolio
        return list(cml_portfolio)

    def weights_to_positions(self, weights, cash):
        # cash must be in USD
        amounts = np.array(weights) * cash
        n_stocks = amounts / [dp[0] for dp in self.stock_prices]
        optimal_positions = {}
        for i, pos in enumerate(self.positions):
            optimal_positions[pos.ticker_symbol] = int(n_stocks[i])
        return optimal_positions

    def get_random_weights(self, min_position_weight=0.01, max_position_weight=0.99):
        if min_position_weight == 0:
            min_position_weight = 0.00000000001  # for some reason weights are not added when the initial weight is 0

        weights = np.full(len(self), min_position_weight)
        assert sum(weights) <= 1.00000000001
        remaining_percentage_total = 1 - sum(weights)
        while remaining_percentage_total > 0:
            add_percentage = np.random.uniform(0, max(remaining_percentage_total, 0.0001))
            weight_index = np.random.randint(len(self))
            if add_percentage > remaining_percentage_total:
                add_percentage = remaining_percentage_total
            if weights[weight_index] + add_percentage > max_position_weight:
                add_percentage = max_position_weight - weights[weight_index]
            remaining_percentage_total -= add_percentage
            weights[weight_index] += add_percentage
        if not 0.99999 < sum(weights) < 1.000001:
            raise RuntimeError(f"sum of weights {weights} does not add up to 100%: sum={sum(weights)}")
        return weights

    def plot_optimal_portfolio(self, min_position_weight=0.01, max_position_weight=0.99, n_portfolios=1000,
                               risk_aversion=1):
        start = time.time()
        fig, ax = plt.subplots()
        self._portfolios = {}

        # plot a few random portfolios
        efficiency_frontier = {}
        for i in range(n_portfolios * len(self)):
            random_weights = self.get_random_weights(min_position_weight=min_position_weight,
                                                     max_position_weight=max_position_weight)
            sd = self.get_standard_deviation(random_weights) * 100
            er = self.get_expected_return(random_weights) * 100
            add = True
            for (sd_efficient, er_efficient) in efficiency_frontier:
                if sd <= sd_efficient and er >= er_efficient:
                    efficiency_frontier.pop((sd_efficient, er_efficient,))
                    break
                elif sd >= sd_efficient and er <= er_efficient:
                    add = False
                    break
            if add:
                efficiency_frontier[(sd, er,)] = random_weights

        for (sd, er), weights in efficiency_frontier.items():
            ax.scatter(sd, er, c="grey")
            self._portfolios[str(sd) + str(er)] = weights

        # plot individual assets
        max_sd = 0
        for pos in self.positions:
            sd = ((pos.standard_deviation) * np.sqrt(252))

            ax.scatter(sd * 100, ((pos.expected_return + 1) ** 365 - 1) * 100, c="red")

            if pos.standard_deviation > max_sd:
                max_sd = pos.standard_deviation

        # plot MVP
        mvp_weights = self.get_mvp_weights(min_position_weight=min_position_weight,
                                           max_position_weight=max_position_weight)
        sd = self.get_standard_deviation(mvp_weights) * 100
        er = self.get_expected_return(mvp_weights) * 100
        ax.scatter(sd, er, c="blue")
        self._portfolios[str(sd) + str(er)] = mvp_weights

        ax.set_xlabel("Standard Deviation (%)")
        ax.set_ylabel("Expected Return (% annually)")
        crs = mplcursors.cursor(ax, hover=True)
        crs.connect("add", self.on_hover)
        print(f"finished plotting portfolio in {int(time.time() - start)} seconds")
        plt.show()

    def _repr_weights(self, weights):
        repr_weights = {}
        for i, pos in enumerate(self.positions):
            repr_weights[pos.ticker_symbol] = str(round(weights[i] * 100, 2)) + "%"
        return repr_weights

    def on_click(self, x, y, button, pressed):
        if button == pynput.mouse.Button.left and pressed:
            self._listener.stop()
            sd, er = self._selected
            try:
                weights = self._portfolios[str(sd) + str(er)]
                print(f"\nWeights for selected portfolio: {self._repr_weights(weights)}\n"
                      f"Standard Deviation: {round(sd, 2)}%, Expected Return: {round(er, 2)}%")

            except KeyError:
                pass
        else:
            self._listener.stop()

    def on_hover(self, point):
        sd, er = point.target[0], point.target[1]
        self._selected = sd, er

        # when the thread is started while already running this trhows a runtimeerror, i cant be bothered to fix this
        self._listener.stop()
        self._listener = Listener(on_click=self.on_click)
        self._listener.start()

        for pos in self.positions:
            if sd / 100 == pos.standard_deviation and er / 100 == (pos.expected_return + 1) ** 365 - 1:
                point.annotation.set_text(f"Company: {pos.ticker_symbol}\n"
                                          f"Expected Return: {round(er, 2)}%\n"
                                          f"Standard Deviation: {round(sd, 2)}%")

                return

        point.annotation.set_text(f"Expected Return: {round(er, 2)}%\n"
                                  f"Standard Deviation: {round(sd, 2)}%")

    def _get_scatter_data(self, weights):
        return {"x": self.get_standard_deviation(weights), "y": self.get_expected_return(weights)}

    def __getitem__(self, i):
        return self.positions[i]

    def __len__(self):
        return len(self.positions)


class Asset:
    # asset sd and er are daily for easier calculations
    def __init__(self, ticker_symbol, portfolio):
        self.ticker_symbol = ticker_symbol
        self.daily_returns = np.array(get_daily_returns(self.ticker_symbol))

        if len(self.daily_returns) <= 365:
            logger.warning(
                f"Asset {self.ticker_symbol} only has data available for {len(self.daily_returns)} days back, calculations with this assets have limited accuracy")
        self.expected_return = np.mean(self.daily_returns)
        self.standard_deviation = np.std(self.daily_returns)
        self.variance = self.standard_deviation ** 2
        self.sharpe_ratio = (
                                        self.expected_return - portfolio.risk_free_rate) / self.standard_deviation  # this approach is pretty naive

    def __repr__(self):
        return f"Asset instance for ticker_symbol {self.ticker_symbol}"
