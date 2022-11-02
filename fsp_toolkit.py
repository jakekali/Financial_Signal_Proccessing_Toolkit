import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import math

''' Obtain stock data using the data from Yahoo Finance. 
- tickers: The list of tickers to be downloaded. 
- start: The start date of the data.
- end: The end date of the data.
- cache: Whether to cache the data or not. If true, the data will be saved to a pickle file and loaded from the pickle file if it exists.

Returns: A dataframe of the stock data.
'''
def get_stock_data(tickers, start, end, cache=True):

    if(cache):
        try:
            stocks= pd.read_pickle(str(start) + str(end) + str(tickers) +'.pkl')
            print("Loaded stocks from pickle")
            return stocks
        except:
            print("Could not load stocks from pickle")

    stocks = pd.DataFrame()
    for tick in tickers:
        # Pull Data From Yahoo Finance
        stock= web.DataReader(tick,'yahoo',start,end).drop(columns=['High','Low','Open','Volume'])
        # Calculate Daily Returns
        stock= stock.pct_change().dropna()
        # Save Data to Large Dataframe
        stocks[tick]= stock[stock.columns[0]]
    if cache:
        stocks.to_pickle(str(start) + str(end) + str(tickers) +'.pkl')
    return stocks


''' Calculates the daily risk free rate of return for the given time period, using US 3MONTH T-Bills.
- start: The start date of the data.
- end: The end date of the data.
- days: The number of trading days in a year.

Returns: A dataframe of the risk free rate of return.
'''
def get_rf_rate(start, end, days=252):
    fed_data= web.DataReader(['TB3SMFFM','FEDFUNDS'],'fred',start,end)
    fed_data['3MO T-BILL'] = fed_data['TB3SMFFM'] + fed_data['FEDFUNDS']
    return (fed_data['3MO T-BILL'].resample(rule='B').ffill().to_frame())/(100*days) # Daily risk-free rate

'''
Calculates the difference between the prediction of single factor model and the actual return.

$ Y = \alpha + \beta*X $
$ residuals = Y - (\alpha + \beta \times X)$

- stocks: The pandas dataframe containing the stock data.
- beta: A list of  beta values for the stocks.
- alpha: A list of alpha values for the stocks.
- X: The ticker of the factor used in the model. This is usually the market portfolio. The default vale is '^GSPC' (S&P 500).
- Y: The list of tickers of the stocks to be used in the model.

Returns: A dataframe of the residuals.
'''
def residualCalculator(stocks, beta, alpha, X='^GSPC', Y=None):

    sys = pd.DataFrame()
    # Calculate Residuals
    # print("beta length: ", len(beta))
    for i, tick in enumerate(Y): 
        sys[tick + ' Residuals']= stocks[tick] - (alpha[i] + beta[i]*stocks[X])

    return sys

''' Calculates a single factor model for the given stocks and factor.
- stocks: The pandas dataframe containing the stock data.
- X: The ticker of the factor used in the model. This is usually the market portfolio. The default vale is '^GSPC' (S&P 500).
- Y: The list of tickers of the stocks to be used in the model.
- cov: The covariance matrix of the stocks. If not provided, the covariance matrix will be calculated from the stocks dataframe.

Returns: A tuple containing the beta, alpha, and residuals.
'''
def singleFactorModel(stocks, X='^GSPC', Y=None, cov=None):
    # Calculate Covariance Matrix
    if cov is None:
        cov= stocks.cov()
    # Calculate Beta
    beta= cov[X]/cov[X][X]
    # Calculate Alpha
    alpha= stocks[Y].mean() - beta*stocks[X].mean()
    # Calculate Residuals
    residuals= residualCalculator(stocks, beta, alpha, X, Y)
    return beta, alpha, residuals


''' Calculates the minium variance portfolio for the given stocks.
- stocks: The pandas dataframe containing the stock data.
- tickers: The list of tickers of the stocks to be used in the model.
- cov: The covariance matrix of the stocks. If not provided, the covariance matrix will be calculated from the stocks dataframe.
- inv_covs: The inverse of the covariance matrix of the stocks. If not provided, the covariance matrix will be calculated from the stocks dataframe.
- m_ex: The expected return of the stocks. If not provided, the expected return will be calculated from the stocks dataframe, by calculating the mean of returns over the period in question.

Returns: A tuple containing the weights of the minium variance portfolio, the variance of the minium variance portfolio, and the expected return of the minium variance portfolio.
'''
def calcMVP(stocks, tickers, cov=None, inv_covs=None, m_ex=None):
    if cov is None:
        covs = stocks[tickers].cov().values
    if inv_covs is None or cov is None:
        cov_stocks_inv = np.linalg.inv(covs)
    if m_ex is None:
        m_ex = stocks[tickers].describe(percentiles=[]).loc['mean']

    denom = (np.ones(stocks[tickers].shape[1]).T @ cov_stocks_inv @ np.ones(stocks[tickers].shape[1]))
    w_mvp = (cov_stocks_inv @ np.ones(stocks[tickers].shape[1]))/denom
    sigma_mvp = 1/math.sqrt(denom)
    mu_mvp = (np.ones(stocks[tickers].shape[1]).T @ cov_stocks_inv @ m_ex) / denom

    return w_mvp, sigma_mvp, mu_mvp

''' Calculates the Market Portfolio for the given stocks.
- stocks: The pandas dataframe containing the stock data.
- tickers: The list of tickers of the stocks to be used in the model.
- cov: The covariance matrix of the stocks. If not provided, the covariance matrix will be calculated from the stocks dataframe.
- inv_covs: The inverse of the covariance matrix of the stocks. If not provided, the covariance matrix will be calculated from the stocks dataframe.
- m_ex: The expected return of the stocks. If not provided, the expected return will be calculated from the stocks dataframe, by calculating the mean of returns over the period in question.

Returns: A tuple containing the weights of the Market Portfolio, the variance of the Market Portfolio, and the expected return of the Market Portfolio.
'''
def calcMP(stocks, tickers, covs=None, cov_stocks_inv=None, m_ex=None):
    if covs is None:
        covs = stocks[tickers].cov().values
    if cov_stocks_inv is None or covs is None:
        cov_stocks_inv = np.linalg.inv(covs)
    if m_ex is None:
        m_ex = stocks[tickers].describe(percentiles=[]).loc['mean']

    # W_MP_S = Weight of the Market Portfolio Scaling Factor
    w_mp_s = 1/(np.ones(stocks[tickers].shape[1]) @ cov_stocks_inv @ m_ex)
    w_mp = w_mp_s * cov_stocks_inv @ m_ex

    # Calculate mu and sigma for the market portfolio
    mu_mp = w_mp @ stocks[tickers].mean()
    sigma_mp = math.sqrt(w_mp @ stocks[tickers].cov() @ w_mp)

    return w_mp, sigma_mp, mu_mp


'''  Calculates the efficient frontier for the given stocks, with no risk free asset. 
- stocks: The pandas dataframe containing the stock data.
- tickers: The list of tickers of the stocks to be used in the model.
start: The starting point of the efficient frontier. The default value is -0.025
end: The ending point of the efficient frontier. The default value is 0.025
step: The number of steps to be included in the efficient frontier. The default value is 2000
- cov: The covariance matrix of the stocks. If not provided, the covariance matrix will be calculated from the stocks dataframe.
- inv_covs: The inverse of the covariance matrix of the stocks. If not provided, the covariance matrix will be calculated from the stocks dataframe.

Returns: A list of sigma values, a list of mu values which correspond to the X,Y coordinates of the efficient frontier
'''
def calcEfficientFrontier(stocks, tickers, start=-0.025, stop=0.025, step=2000, cov=None, inv_covs=None):

    sigma_ef = []
    mu_ef = np.linspace(-0.025, 0.025, 2000)
    mus = stocks[tickers].mean().values
    if cov is None:
        covs = stocks[tickers].cov().values
    if inv_covs is None or cov is None:
        inv_covs = np.linalg.inv(covs)

    for mu in mu_ef:
        mu_t = np.array([mu,1])
        m_t = np.concatenate([mus.reshape(len(mus),1), np.ones((len(mus),1))], axis=1)
        B = m_t.T  @ inv_covs @ m_t
        sigma_v = (mu_t.T  @ np.linalg.inv(B))
        sigma_v = sigma_v @ m_t.T 
        sigma_v = sigma_v @ inv_covs 
        sigma_v = sigma_v @ m_t 
        sigma_v = sigma_v @ np.linalg.inv(B) 
        sigma_v = sigma_v @ mu_t
        sigma_v = np.sqrt(sigma_v)
        sigma_ef.append(sigma_v)

    return sigma_ef, mu_ef

''' Plots the efficient frontier for the given stocks, both with and without a risk free asset.
- efficientFrontier: The efficient frontier to be plotted. (The output of calcEfficientFrontier)
- MP: The Market Portfolio to be plotted. (The output of calcMP)
- MVP: The Minimum Variance Portfolio to be plotted. (The output of calcMVP)
- ax: The axis to be plotted on. If not provided, a new axis will be created.

Returns: The axis that was plotted on, with the efficient frontier plotted on it.
'''
def plotEfficientFrontier(efficientFrontier, MP, MVP, ax=None):
    if(ax is None):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

    ax.plot(efficientFrontier[0], efficientFrontier[1], label='Efficient Frontier')

    slope = MP[1]/MP[0]

    ax.plot([0, max(efficientFrontier[0])],[0,slope*max(efficientFrontier[0])], label='CML')
    ax.scatter(MP[0], MP[1], label='Market Portfolio', color='red')
    ax.scatter(MVP[0], MVP[1], label='Minimum Variance Portfolio', color='green')
    ax.scatter(0, 0, label='Risk Free Asset', color='black')

    ax.set_xlabel('Standard Deviation')
    ax.set_ylabel('Expected Return')

    ax.legend()

    return ax


# # Now Let's Calculate the Market Portfolio
# import math

# tickers = te
# cov_stocks_inv = np.linalg.inv(stocks[tickers].cov())

# # Mean Expected Returns
# m_ex = stocks[tickers].describe(percentiles=[]).loc['mean']

# # W_MP_S = Weight of the Market Portfolio Scaling Factor
# w_mp_s = 1/(np.ones(stocks[tickers].shape[1]) @ cov_stocks_inv @ m_ex)
# w_mp = w_mp_s * cov_stocks_inv @ m_ex

# weights_show = pd.DataFrame([w_mp, m_ex], columns=tickers, index=['Weight of the Market Portfolio', 'Mean Expected Returns'])
# display(weights_show.head(3))

# # Calculate mu and sigma for the market portfolio
# mu_mp = w_mp @ stocks[tickers].mean()
# sigma_mp = math.sqrt(w_mp @ stocks[tickers].cov() @ w_mp)
# print("The mean of the market portfolio is ", round(mu_mp,6), "and the standard deviation is: ", round(sigma_mp,6))

# mu_mp = (m_ex.T @ cov_stocks_inv @ m_ex) * w_mp_s
# sigma_mp = np.sqrt(m_ex.T @ cov_stocks_inv @ m_ex) * w_mp_s

# print("The MP lies on the point (", round(sigma_mp,6),",", round(mu_mp,6),")")