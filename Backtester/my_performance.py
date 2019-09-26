from __future__ import print_function
import numpy as np
import pandas as pd


def return_by_period(returns, period='year'):
        '''
        arranges data based on different time frame 
        '''
        if period == 'year':
            returns[period] = [s.year for s in returns.index]
            returns = returns.groupby('year').apply(lambda data: np.cumprod(1+data['returns']).iloc[-1]-1)
        elif period == 'month':
            returns['year'] = [s.year for s in returns.index]
            returns['month'] = [s.month for s in returns.index]
            returns = returns.set_index(['year', 'month'])
            returns = returns.groupby(['year','month']).apply(lambda data: np.cumprod(1+data['returns']).iloc[-1]-1)
        return returns


def create_returns_and_sharpe_ratio(returns, periods=252):
    """
    Create the Returns and Sharpe ratio for the strategy, based on a 
    benchmark of zero (i.e. no risk-free rate information).

    Parameters:
    returns - A pandas Series representing period percentage returns.
    periods - Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    """
    
    tot_ret = np.cumprod(1+returns).iloc[-1]-1
    years = len(returns)/periods
    cagr = (tot_ret+1)**(1/years)-1
    anul_std = returns.std()*np.sqrt(periods)
    sharpe = cagr / anul_std
    return tot_ret[0], cagr[0], anul_std[0], sharpe[0]
    

def create_drawdowns(pnl):
    """
    Calculate the largest peak-to-trough drawdown of the PnL curve
    as well as the duration of the drawdown. Requires that the 
    pnl_returns is a pandas Series.

    Parameters:
    pnl - A pandas Series representing period percentage returns.

    Returns:
    drawdown, duration - Highest peak-to-trough drawdown and duration.
    """

    # Calculate the cumulative returns curve 
    # and set up the High Water Mark
    # Then create the drawdown and duration series
    hwm = [0]
    eq_idx = pnl.index
    drawdown = pd.Series(index = eq_idx)
    duration = pd.Series(index = eq_idx)

    # Loop over the index range
    for t in range(1, len(eq_idx)):
        cur_hwm = max(hwm[t-1], pnl[t])
        hwm.append(cur_hwm)
        drawdown[t]= (hwm[t] - pnl[t]) / hwm[t]
        duration[t]= (0 if drawdown[t] == 0 else duration[t-1] + 1)
    return drawdown, drawdown.max(), duration.max()