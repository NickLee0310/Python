from __future__ import print_function
import datetime
import pprint
try:
    import Queue as queue
except ImportError:
    import queue
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from my_performance import create_returns_and_sharpe_ratio, create_drawdowns, return_by_period
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#sns.set_palette('Set2')
#plt.style.use('ggplot')

class Backtest(object):
    """
    Enscapsulates the settings and components for carrying out
    an event-driven backtest.
    """
    def __init__(
        self, csv_dir, symbol_list, initial_capital,
        heartbeat, start_date, end_date, data_handler,
        execution_handler, portfolio, strategy
        ):
        """
        Initialises the backtest.
        Parameters:
        csv_dir - The hard root to the CSV data directory.
        symbol_list - The list of symbol strings.
        intial_capital - The starting capital for the portfolio.
        heartbeat - Backtest "heartbeat" in seconds
        start_date - The start datetime of the strategy.
        data_handler - (Class) Handles the market data feed.
        execution_handler - (Class) Handles the orders/fills for trades.
        portfolio - (Class) Keeps track of portfolio current
        and prior positions.
        strategy - (Class) Generates signals based on market data.
        """
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date
        self.end_date = end_date
        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        self.events = queue.Queue()
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1
        self._generate_trading_instances()
        
    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from
        their class types.
        """
        print(
        "Creating DataHandler, Strategy, Portfolio and ExecutionHandler"
        )
        self.data_handler = self.data_handler_cls(self.events, self.csv_dir,
        self.symbol_list, self.start_date)
        self.strategy = self.strategy_cls(self.data_handler, self.events)
        self.portfolio = self.portfolio_cls(self.data_handler, self.events,
                                            self.start_date,
                                            self.initial_capital)
        self.execution_handler = self.execution_handler_cls(self.events)
    def _run_backtest(self):
        """
        Executes the backtest.
        """
        i = 0
        while True:
            i += 1
            #print (i)
            # Update the market bars
            if self.data_handler.continue_backtest == True:
                self.data_handler.update_bars()
            else:
                break
           #print(self.data_handler.current_date)
           
            # Check if current date is in the period    
            if datetime.datetime.strptime(self.data_handler.current_date, '%Y-%m-%d') > self.end_date:
                break
                
            # Handle the events
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.type == 'MARKET':                                  
                            self.strategy.calculate_signals(event)                      
                            self.portfolio.update_timeindex(event)                       
                        elif event.type == 'SIGNAL':                    
                            self.signals += 1
                            self.portfolio.update_signal(event)
                        elif event.type == 'ORDER':               
                            self.orders += 1
                            self.execution_handler.execute_order(event)
                        elif event.type == 'FILL':             
                            self.fills += 1
                            self.portfolio.update_fill(event)
                                                        
            time.sleep(self.heartbeat)
            
    def plot_results(self):  
        '''
        Plot the results
        '''
        plt.figure(figsize = (12, 12))
        rect1 = [0.14, 0.90, 0.9, 0.05] 
        rect2 = [0.14, 0.55, 0.9, 0.30]
        rect3 = [0.14, 0.33, 0.9, 0.15]
       
        returns = self.portfolio.equity_curve[['returns']]
        pnl = self.portfolio.equity_curve['equity_curve']
        tot_ret, cagr, std, sharpe = create_returns_and_sharpe_ratio(returns)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
       
                  
        # plot performance
        ax1 = plt.axes(rect1)
        ax1.text(0.03, 0.7, 'Total Return', fontsize=13)
        ax1.text(0.07, 0.2, '{:.0%}'.format(tot_ret), fontsize=12)
        ax1.text(0.19, 0.7, 'CAGR', fontsize=13)
        ax1.text(0.20, 0.2, '{:.1%}'.format(cagr), fontsize=12)
        ax1.text(0.30, 0.7, 'Annual Volatility', fontsize=13)
        ax1.text(0.35, 0.2, '{:.1%}'.format(std), fontsize=12)
        ax1.text(0.48, 0.7, 'Sharp Ratio', fontsize=13)
        ax1.text(0.51, 0.2, '{:.1f}'.format(sharpe), fontsize=12)
        ax1.text(0.62, 0.7, 'Max Drawdown', fontsize=13)
        ax1.text(0.67, 0.2, '{:.1%}'.format(max_dd), fontsize=12)
        ax1.text(0.78, 0.7, 'Max Drawdown Duration', fontsize=13)
        ax1.text(0.87, 0.2, '{:.0f}'.format(dd_duration), fontsize=12)
        ax1.grid(False)
        ax1.spines['top'].set_linewidth(2.0)
        ax1.spines['bottom'].set_linewidth(2.0)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        ax1.set_ylabel('')
        ax1.set_xlabel('')
        ax1.set_title('Results', fontweight='bold', fontsize=15)

        # plot curve 
        ax2 = plt.axes(rect2)
        ax2.plot(self.portfolio.equity_curve['equity_curve'], color='b')
        ax2.set_title('Performance', fontsize=15)
        ax2.legend(['Strategy'], loc='upper left',  fontsize=15)
        ax2.hlines(1, linestyles='--', xmin=self.start_date, xmax=self.end_date)
        ax2.set_ylabel('Cumulative returns')
        ax2.grid()
        
        # plot drawdown
        ax3 = plt.axes(rect3)
        ax3.set_title('Drawdown(%)', fontsize=15)
        ax3.plot(-self.portfolio.equity_curve['drawdown'], color='r')
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax3.grid()       
        plt.show()
    
    def plot_yearly_returns(self):
        '''
        Plot returns by year
        '''
        returns  = self.portfolio.equity_curve[['returns']]
        yearly_return = return_by_period(returns, period='year')
        plt.figure(figsize = (12, 6))
        ax = plt.gca()
        ax.set_title('Yearly Returns(%)', fontsize=15)
        ax.bar(yearly_return.index, yearly_return, color='gray')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.set_xticklabels([int(x) for x in ax.get_xticks()], rotation=45)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid()
        plt.show()
          
    def plot_monthly_returns(self):
        '''
        Plot returns by month
        '''
        returns  = self.portfolio.equity_curve[['returns']]
        monthly_return = return_by_period(returns, period='month')
        monthly_return = monthly_return.unstack()
        plt.figure(figsize = (12, 6))
        ax = plt.gca()      
        monthly_return = np.round(monthly_return, 3)
        monthly_return.rename(
            columns={1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
                     5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                     9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'},
            inplace=True
        )
        sns.heatmap(
                    monthly_return.fillna(0) * 100.0,
                    annot=True,
                    fmt="0.1f",
                    annot_kws={"size": 12},
                    alpha=1.0,
                    center=0.0,
                    cbar=False,
                    cmap=cm.RdYlGn,
                    ax=ax)
        ax.set_title('Monthly Returns (%)', fontsize=15)
        ax.set_ylabel('')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel('')
        plt.show()
                       
    def simulate_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        print('Backtesting start')
        start = time.time()
        self._run_backtest()
        print('Backtesting over! Comsuming %.2f s' % (time.time()-start))
        self.portfolio.create_equity_curve_dataframe()                                              
                
