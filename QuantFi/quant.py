import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
import numpy as np
import ccxt
import cvxpy as cp
import yfinance as yf
from empyrical import max_drawdown, calmar_ratio, sortino_ratio
from PyPortfolioOpt.pypfopt.hierarchical_portfolio import HRPOpt
from PyPortfolioOpt.pypfopt import risk_models, expected_returns
from PyPortfolioOpt.pypfopt import expected_returns, risk_models, EfficientFrontier




class Portfolio:
    def __init__(self, ticker_list, start_date, end_date):
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = self._download_data()
        self.returns = np.log(self.price_data/self.price_data.shift()).dropna()
        self.weights = None
        self.ef = None
        
    def _download_data(self):
        prices = yf.download(self.ticker_list, start=self.start_date, end=self.end_date)['Adj Close']
        prices = prices.dropna()
        return prices

    def compute_ef(self, returns):
        mean = expected_returns.mean_historical_return(returns)
        std = risk_models.sample_cov(returns)
        ef = EfficientFrontier(mean, std)
        self.ef = ef

    def max_sharpe(self, verbose=False):
        weights = self.ef.max_sharpe()
        self.ef.portfolio_performance(verbose=verbose)
        return weights

    def min_volatility(self, verbose=False):
        weights = self.ef.min_volatility()
        self.ef.portfolio_performance(verbose=verbose)
        return weights
    
    def hrp_allocation(self, returns, verbose=False):
        hrp = HRPOpt(returns)
        weights = hrp.optimize()
        hrp.portfolio_performance(verbose=verbose)
        return weights
    
    def ERC_allocation(self, returns):
        # function to calculate portfolio risk
        def calculate_portfolio_risk(weights, cov_matrix):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            return np.sqrt(portfolio_variance)

        # function to calculate asset contribution to total risk
        def calculate_risk_contribution(weights, cov_matrix):
            portfolio_risk = calculate_portfolio_risk(weights, cov_matrix)
            # Marginal Risk Contribution
            MRC = np.dot(cov_matrix, weights) / portfolio_risk
            # Risk Contribution
            RC = weights * MRC
            return RC

        # objective function to minimize
        # (difference between each asset's contribution to risk and the average risk contribution)
        def objective_function(weights, cov_matrix):
            avg_risk_contribution = 1.0 / len(weights)
            risk_contributions = calculate_risk_contribution(weights, cov_matrix)
            return np.sum((risk_contributions - avg_risk_contribution)**2)
        
        # compute the covariance matrix
        cov_matrix = returns.cov().values  # converting to numpy array for use in functions

        # number of assets
        n = len(cov_matrix)

        # constraints (weights must sum to 1)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})

        # bounds (weights must be between 0 and 1)
        bounds = tuple((0.05,0.45) for asset in range(n))

        # initial guess (equal weights)
        weights_init = np.array([1 / 2*n] * n)

        # minimize the objective function
        result = minimize(objective_function, weights_init, args=cov_matrix, method='SLSQP', constraints=constraints, bounds=bounds)

        weights_erc = dict(zip(returns.columns.values, result.x))

        return weights_erc
    
    def compute_metrics(self, weights, VERBOSE=True, rebalance_freq=""):
        #metric_names = ["Average Annual Return", "STD Returns", "Sharpe Ratio", "Sortino Ratio", "Maximum Drawdown", "Calmar Ratio"]
        if rebalance_freq == "":
            portfolio_returns = (weights* self.returns).sum(axis=1)

        else:
            periods = pd.date_range(start=self.start_date, end=self.end_date, freq=rebalance_freq)
            portfolio_return = pd.DataFrame(dtype=float)

            for i in range(len(periods) - 1):
                start, end = periods[i], periods[i+1]
                period_returns = self.returns.loc[start:end]
                # Compute portfolio returns
                portfolio_temp = (weights[i] * period_returns)
                portfolio_return = portfolio_return.append(portfolio_temp)

            # total_compounded_log_return
            portfolio_returns = portfolio_return.sum(axis=1)

        # Cummulative return
        total_growth_factor = np.exp(portfolio_returns) - 1

        # Average annual return
        average_annual_return = portfolio_returns.mean() * 252

        # Standard deviation of returns
        std_dev_returns = portfolio_returns.std()

        # Sharpe ratio (assuming a risk-free rate of 0)
        sharpe_ratio = average_annual_return / std_dev_returns

        # Sortino ratio
        sortino = sortino_ratio(portfolio_returns)

        # Max drawdown
        maximum_drawdown = max_drawdown(portfolio_returns)

        # Calmar ratio
        calmar = calmar_ratio(portfolio_returns)

        if VERBOSE:
            print(F"\nPeriod: {self.start_date} - {self.end_date}")
            print(f'Weights: {weights[3]}')
            print(f'Cumulative returns: {total_growth_factor.sum()*100}')
            print(f"Average Annual Return: {average_annual_return*100}")
            print(f"STD Returns: {std_dev_returns*100}")
            print(f"Sharpe Ratio: {sharpe_ratio}")
            print(f"Sortino Ratio: {sortino}")
            print(f"Maximum Drawdown: {maximum_drawdown*100}")
            print(f"Calmar Ratio: {calmar}")

    def _rebalance(self, period_returns, model):
            
        if model == "max_sharpe":
            self.compute_ef(period_returns)
            return self.ef.max_sharpe()

        elif model == "min_vol":
            self.compute_ef(period_returns)
            return self.ef.min_volatility()
            
        elif model == "hrp":
            return self.hrp_allocation(period_returns)
        
        elif model == "erc":
            return self.ERC_allocation(period_returns)
        
    def compute_portfolio(self, model, rebalance_freq=''):
        self.weights = []

        if rebalance_freq != '':
            periods = pd.date_range(start=self.start_date, end=self.end_date, freq=rebalance_freq)

            for i in range(len(periods) - 1):
                start, end = periods[i], periods[i+1]
                period_returns = self.returns.loc[start:end]
                weights = self._rebalance(period_returns, model)
                self.weights.append(weights)
        else:
            weights = self._rebalance(self.returns, model)
            self.weights.append(weights)

        return self.weights


def main():
    portfolio = Portfolio(["BTC-USD","DBE", "ARKK", "GOLD.AX"], start_date='2020-01-01', end_date='2022-12-31')
    weights = portfolio.compute_portfolio(model="max_sharpe", rebalance_freq="")
   # portfolio.compute_metrics(weights,  rebalance_freq='')  
    # weights = portfolio.hrp_allocation(portfolio.returns, verbose=True)
    # print(weights)
    # portfolio.compute_metrics(weights)
    
if __name__ == "__main__":
    main()