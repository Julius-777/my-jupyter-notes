#
# Python Module with Class
# for Vectorized Backtesting
# of SMA-based Strategies
#``
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import numpy as np
import pandas as pd
from scipy.optimize import brute
import yfinance as yf
import ta
import matplotlib.pyplot as plt


# class SMAVectorBacktester(object):
#     ''' Class for the vectorized backtesting of SMA-based trading strategies.

#     Attributes
#     ==========
#     symbol: str
#         RIC symbol with which to work with
#     SMA1: int
#         time window in days for shorter SMA
#     SMA2: int
#         time window in days for longer SMA
#     start: str
#         start date for data retrieval
#     end: str
#         end date for data retrieval

#     Methods
#     =======
#     get_data:
#         retrieves and prepares the base data set
#     set_parameters:
#         sets one or two new SMA parameters
#     run_strategy:
#         runs the backtest for the SMA-based strategy
#     plot_results:
#         plots the performance of the strategy compared to the symbol
#     update_and_run:
#         updates SMA parameters and returns the (negative) absolute performance
#     optimize_parameters:
#         implements a brute force optimizeation for the two SMA parameters
#     '''

#     def __init__(self, symbol, SMA1, SMA2, start, end):
#         self.symbol = symbol
#         self.SMA1 = SMA1
#         self.SMA2 = SMA2
#         self.start = start
#         self.end = end
#         self.results = None
#         self.get_data()

#     def get_data(self):
#         ''' Retrieves and prepares the data.
#         '''
#         try:
#             raw = yf.download(self.symbol, start=self.start, end=self.end)['Adj Close'].dropna(how="any")
#         except Exception as e:
#             # Handle the exception here
#             print("An error occurred while downloading data:", e)

#         raw = pd.DataFrame(raw)
#         raw.rename(columns={'Adj Close': 'price'}, inplace=True)
#         raw['return'] = np.log(raw / raw.shift(1))
#         raw['SMA1'] = raw['price'].rolling(self.SMA1).mean()
#         raw['SMA2'] = raw['price'].rolling(self.SMA2).mean()
#         self.data = raw

#     def set_parameters(self, SMA1=None, SMA2=None):
#         ''' Updates SMA parameters and resp. time series.
#         '''
#         if SMA1 is not None:
#             self.SMA1 = SMA1
#             self.data['SMA1'] = self.data['price'].rolling(
#                 self.SMA1).mean()
#         if SMA2 is not None:
#             self.SMA2 = SMA2
#             self.data['SMA2'] = self.data['price'].rolling(
#                 self.SMA2).mean()

#     def run_strategy(self):
#         ''' Backtests the trading strategy.
#         '''
#         data = self.data.copy().dropna()
#         data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
#         data['strategy'] = data['position'].shift(1) * data['return']
#         data.dropna(inplace=True)
#         data['creturns'] = data['return'].cumsum().apply(np.exp)
#         data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
#         self.results = data
#         # gross performance of the strategy
#         aperf = data['cstrategy'].iloc[-1]
#         # out-/underperformance of strategy
#         operf = aperf - data['creturns'].iloc[-1]
#         return round(aperf, 2), round(operf, 2)

#     def plot_results(self):
#         ''' Plots the cumulative performance of the trading strategy
#         compared to the symbol.
#         '''
#         if self.results is None:
#             print('No results to plot yet. Run a strategy.')
#         title = '%s | SMA1=%d, SMA2=%d' % (self.symbol,
#                                                self.SMA1, self.SMA2)
#         self.results[['creturns', 'cstrategy']].plot(title=title,
#                                                      figsize=(10, 6))

#     def update_and_run(self, SMA):
#         ''' Updates SMA parameters and returns negative absolute performance
#         (for minimazation algorithm).

#         Parameters
#         ==========
#         SMA: tuple
#             SMA parameter tuple
#         '''
#         self.set_parameters(int(SMA[0]), int(SMA[1]))
#         return -self.run_strategy()[0]

#     def optimize_parameters(self, SMA1_range, SMA2_range):
#         ''' Finds global maximum given the SMA parameter ranges.

#         Parameters
#         ==========
#         SMA1_range, SMA2_range: tuple
#             tuples of the form (start, end, step size)
#         '''
#         opt = brute(self.update_and_run, (SMA1_range, SMA2_range), finish=None)
#         return opt, -self.update_and_run(opt)

class GeneralBacktester():
    def __init__(self, symbol, start, end, window1, window2=200, strategy='SMA1_SMA2', 
                 short=False, tcost=0.0, tax_rate=0.3, atr_window=14, atr_multiplier=3):
        self.symbol = symbol
        self.window1 = window1
        self.window2 = window2
        self.start = start
        self.end = end
        self.strategy = strategy
        self.tcost = tcost
        self.tax_rate = tax_rate
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
        self.results = None
        self.short = short
        self.get_data()

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        try:
            raw = yf.download(self.symbol, start=self.start, end=self.end)[['High','Low', 'Adj Close']].dropna(how="any")
        except Exception as e:
            # Handle the exception here
            print("An error occurred while downloading data:", e)

        raw = pd.DataFrame(raw)
        raw.rename(columns={'Adj Close': 'price'}, inplace=True)
        raw['return'] = np.log(raw['price'] / raw['price'].shift(1)).dropna()
        self.data = raw

    def set_parameters(self, window1=None, window2=200):
        ''' Updates window parameters and resp. time series. '''
        if window1 is not None:
            self.window1 = window1
        if window2 is not None:
            self.window2 = window2
                
    def EMA(self, window1, window2):
        ''' Calculates EMA for given window. '''
        EMA1 = self.data['price'].ewm(span=self.window1).mean()
        EMA2 = self.data['price'].ewm(span=self.window2).mean()
        return EMA1, EMA2

    def RSI(self, window=14, periods=14):
        ''' Calculates RSI for given periods using the `ta` library. '''
        RSI = ta.momentum.RSIIndicator(self.data['price'], periods).rsi()
        SMA_RSI = RSI.rolling(self.window).mean()
        return RSI, SMA_RSI
   
    def SMA(self, window1, window2):
        ''' Calculates SMA for given window. '''
        SMA1 = self.data['price'].rolling(self.window1).mean()
        SMA2 = self.data['price'].rolling(self.window2).mean()     
        return SMA1, SMA2
    
    def ATR(self, data):
        # Calculate ATR Trailing Stop
        data['ATR'] = self.atr_multiplier*ta.volatility.average_true_range(data['High'], data['Low'], data['price'], self.atr_window)
        
        # Set initial trailing stop at entry price - ATR
        data['trailing_stop'] = data['price'] - data['ATR']

        # Identify the rows where a new position is entered
        new_positions = data['position'].diff() != 0

        # For each position, calculate the cumulative maximum of the trailing stop
        data['trailing_stop'] = data.groupby(new_positions.cumsum()).trailing_stop.cummax()


    def run_strategy(self):
        ''' Backtests the specified trading strategy.
        '''
        data = self.data.copy().dropna()
        n = -1 if self.short else 0

        if self.strategy == "SMA1_SMA2":
            data['SMA1'], data['SMA2'] = self.SMA(self.window1, self.window2)
            data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, n)

        elif self.strategy == "EMA_SMA":
            data['EMA1'], data['EMA2']  = self.EMA(self.window1, self.window2)
            data['SMA1'], data['SMA2'] = self.SMA(self.window1, self.window2)
            data['position'] = np.where((data['EMA1'] > data['SMA2']) & (data['price'] > data['EMA1']), 1, n)

        elif self.strategy == "EMA_EMA":
            data['EMA1'], data['EMA2'] = self.EMA(self.window1, self.window2)
            data['position'] = np.where(data['EMA1'] > data['EMA2'], 1, n)

        elif self.strategy in ["EMA", "EMA_ATR"]:
            data['EMA'] = self.EMA(self.window1, self.window2)[0]
            data['position'] = np.where(data['price'] > data['EMA'], 1, n)

        elif self.strategy == "RSI_SMA":
            data['RSI'], data['SMA_RSI'] = self.RSI()
            data['position'] = np.where(data['RSI'] > data['SMA_RSI'], 1, n)
        
        else:
            raise ValueError("Invalid strategy. Choose either 'EMA', 'SMA1_SMA2', 'EMA_SMA', or 'RSI_SMA'.")

        # Add ATR trailing Stop (Optional)
        if self.strategy in ["EMA_ATR", "ATR"]:
            self.ATR(data)
            # Close position when trailing stop is hit
            data['position'] = np.where(data['price'] < data['trailing_stop'], 0, data['position'])

        # Calcuate strategy's trade executions
        data['strategy'] = data['position'].shift(1) * data['return']
        # Add transaction costs
        data['trades'] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy - data.trades * self.tcost
        data.dropna(inplace=True)

        # Calculate cummlative returns for strategy vs asset
        data['creturns'] = data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data

        # Apply after Tax Aus laws
        self.apply_tax()
        
        # Return results of asset vs strategy performance
        aperf = data['cstrategy'].iloc[-1]
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)
    
    def apply_tax(self):
        '''Applies the tax rate to the strategy's returns, considering the 50% discount
        for positions held longer than a year at the time of selling.'''

        data = self.results

        # Determine when a new position is entered
        data['new_position'] = data['position'].diff().fillna(0).abs()

        # Keep track of the date a new position is entered
        data['entry_date'] = np.where(data['new_position'] != 0, data.index, np.datetime64('NaT'))

        # Forward fill entry dates
        data['entry_date'] = data['entry_date'].replace(np.datetime64('NaT')).ffill()

        # Calculate holding period for each position
        data['holding_period'] = data.index - data['entry_date']

        # Determine if each position has been held for over a year
        data['over_year'] = (data['holding_period'] > pd.Timedelta(days=365))

        # Apply tax rate, considering discount for positions held over a year
        data['tax'] = np.where(data['over_year'], self.tax_rate * 0.5 * data['strategy'], 
                            self.tax_rate * data['strategy'])

        data['strategy_after_tax'] = data['strategy'] - data['tax']
        data['cstrategy_after_tax'] = data['strategy_after_tax'].cumsum().apply(np.exp)
    
        return data

    def update_and_run(self, windows):
        ''' Updates SMA/EMA/RSI parameters and returns negative absolute performance
        (for minimization algorithm).

        Parameters
        ==========
        windows: tuple
            window1 and window2 parameter tuple
        '''
        self.set_parameters(int(windows[0]), int(windows[1]))
        return -self.run_strategy()[0]

    def plot_results(self, start=None, end=None, after_tax=False):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        else:
            # Plotting the cumulative returns of the strategy vs. the asset
            title = '%s | Strategy - %s | wind1=%d, wind2=%d' % (self.symbol, self.strategy, self.window1, self.window2)
            plt.figure(figsize=(14,7))
            plt.subplot(2,1,1)
            # Set Plot range
            
            start = self.results.index.min() if start == None else start
            end = self.results.index.max() if end == None else end
 
            strategy = 'cstrategy_after_tax' if after_tax else 'cstrategy'
            self.results[['creturns', strategy]][start:end].plot(title=title, ax=plt.gca())
            plt.ylabel('Cumulative Returns')

            # Plotting the strategy indicators over the price
            plt.subplot(2,1,2)
            self.results['price'][start:end].plot(ax=plt.gca(), alpha=0.5)
            plt.ylabel('Price')

            if self.strategy == 'SMA1_SMA2':
                self.results[['SMA1', 'SMA2']][start:end].plot(ax=plt.gca())

            elif self.strategy == 'EMA_SMA':
                self.results[['EMA1', 'SMA2']][start:end].plot(ax=plt.gca())

            elif self.strategy == 'RSI_SMA':
                self.results[['RSI', 'SMA_RSI']][start:end].plot(ax=plt.gca())

            elif self.strategy == 'EMA':
                self.results['EMA'][start:end].plot(ax=plt.gca())

            elif self.strategy in ["EMA_ATR", "ATR"]:
                self.results['EMA'][start:end].plot(ax=plt.gca())
                self.results['trailing_stop'][start:end].plot(ax=plt.gca())

            else:
                raise ValueError('Invalid strategy provided for plotting')
            
            plt.show()

    def optimize_parameters(self, range_1, range_2):
        ''' Finds global maximum given the window parameter ranges.

        Parameters
        ==========
        range_1, range_2: tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (range_1, range_2), finish=None)
        return opt, -self.update_and_run(opt)


if __name__ == '__main__':
    # smabt = SMAVectorBacktester('BTC-USD', 50, 200,
    #                             '2010-1-1', '2020-12-31')

    # Create a GeneralBacktester object with EMA_SMA strategy
    smabt = GeneralBacktester('BTC-USD','2015-05-1', '2016-09-01', window1=50, window2=200, strategy='EMA_ATR')
    
    # Running strategy with initial parameters
    strategy_results = smabt.run_strategy()
    print("Date Range:",smabt.data.index[0], smabt.data.index[-1],"\n")
    print(f'Gross Strategy {smabt.strategy} Performance with initial parameters (wind1={smabt.window1}, wind2={smabt.window2}): {strategy_results[0]*100}%')
    print(f'Out-/Underperformance with initial parameters (wind1={smabt.window1}, wind2={smabt.window2}): {strategy_results[1]*100}%\n')
    smabt.plot_results(start="2015", end="2016")
    # # Setting new parameters
    # smabt.set_parameters(SMA1=20, SMA2=100)

    # # Running strategy with new parameters
    # strategy_results = smabt.run_strategy()
    # print(f'Gross Strategy Performance with initial parameters (wind1={smabt.window1}, wind2={smabt.window2}): {strategy_results[0]*100}%')
    # print(f'Out-/Underperformance with initial parameters (wind1={smabt.window1}, wind2={smabt.window2}): {strategy_results[1]*100}%\n')
    
    # Optimizing parameters
    opt_results = smabt.optimize_parameters((21, 200, 4), (21, 200, 4))
    print(f'{smabt.strategy} Optimal Parameters: wind1={opt_results[0][0]}, wind2={opt_results[0][1]}')
    print(f'Max Performance with optimal parameters: {opt_results[1]*100}%\n')
    print(f'Asset Performance:', smabt.results['creturns'][-1]*100,"%")

