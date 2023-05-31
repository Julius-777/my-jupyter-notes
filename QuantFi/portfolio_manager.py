import numpy as np
import pandas as pd
import yfinance as yf
from pyPortfolioAnalysis import * 
import scipy.stats as scs

class PortfolioManager:
    def __init__(self, assets):
        self.assets = assets
        self.data = None
        self.weights = None
        self.returns = None

    def fetch_data(self, start_date='2010-01-01', end_date='2023-01-01'):
        self.data = yf.download(self.assets, start=start_date, end=end_date)['Adj Close']
        return self.data

    def calculate_returns(self, type=""):
        if type == "log":
            return np.log(self.data/self.data.shift(1)).dropna()
        
        return self.data.pct_change().dropna()

    def calculate_log_returns(self):
        return np.log(self.data/self.data.shift(1)).dropna()

    def set_weights(self, weights):
        if len(weights) != len(self.assets):
            raise ValueError("Number of weights must equal number of assets.")
        self.weights = np.array(weights)

    def portfolio_returns(self, type=""):
        if self.weights is None:
            raise ValueError("Portfolio weights are not set.")
        
        rets = self.calculate_returns(type=type)
        port_rets = pd.Series(np.dot(rets, self.weights), index=rets.index)
        return port_rets

    def value_at_risk(self, confidence_level=0, verbose=True):
        port_rets = self.portfolio_returns(type="log")

        if confidence_level > 0:
            port_rets = self.portfolio_returns(type="log")
            VaR = scs.norm.ppf(confidence_level, np.mean(port_rets), np.std(port_rets))
            if verbose:
                print("Value at Risk: ", VaR*100, "%\n")
            return VaR
        
        else:
            port_rets = self.portfolio_returns(type="")
            percs = [0.01, 0.1, 1., 2.5, 5.0, 10.0]
            VaR = scs.scoreatpercentile(port_rets, percs)
            if verbose:
                print('{}    {}'.format('Confidence Level', 'Value-at-Risk'))
                print(33 * '-')
                for pair in zip(percs, VaR):
                    print('{:16.2f} {:16.3f}{}'.format(100 - pair[0], -pair[1]*100, "%"))
            return VaR


    def portfolio_simulation(self, portfolio_objective = "min variance"):
        returns = self.calculate_returns()
        # Construct portfolio
        port = portfolio_spec(assets = self.assets)
        # Add Constraints to the portfolio object
        add_constraint(port,kind="long_only")
        add_constraint(port,kind="full_investment")
        # Optimization
        optimize_portfolio(returns, port, optimize_on = portfolio_objective)
        # Print results
        print(port.port_summary())

    def plot_data(self, graph=""):
        if type(self.data) != pd.DataFrame:
            print("load data")
        elif graph == "returns":
            self.calculate_returns().cumsum().plot()

        elif graph == "log_returns":
            self.calculate_log_returns().cumsum().plot()

        elif graph == "port_returns":
            self.portfolio_returns().cumsum().plot()

        else:
            self.data.plot()

def main():
    assets = ["BTC-USD", "DBE", "GOLD.AX", "ARKK"]
    portfolio_manager = PortfolioManager(assets)
    portfolio_manager.fetch_data()
    portfolio_manager.set_weights([0.25, 0.25, 0.25, 0.25])  # weights for each asset
    log_returns = portfolio_manager.calculate_log_returns()
    portfolio_manager.portfolio_returns()
    print("Value at Risk: ", portfolio_manager.value_at_risk()*100, "%")
    print("\n")
    portfolio_manager.portfolio_simulation()

def pyfolio_test():
    assets = ["BTC-USD", "DBE", "GOLD.AX", "ARKK"]
    import pyfolio as pf
    portfolio_manager = PortfolioManager(assets)
    portfolio_manager.fetch_data()
    portfolio_manager.set_weights([0.25, 0.25, 0.25, 0.25])  # weights for each asset
    benchmark_rets = portfolio_manager.calculate_returns()["BTC-USD"]
    port_rets = portfolio_manager.portfolio_returns()

    pf.create_returns_tear_sheet(port_rets, benchmark_rets=benchmark_rets)


if __name__ == "__main__":
    #main()
    pyfolio_test()