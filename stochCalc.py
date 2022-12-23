import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SDE:
    def __init__(X0 : float) -> None:
        '''
        Define an SDE Object, which simulates an SDE of the form: 

        $$X[n] = X[0] + \sum_{m=0}^{n-1} \Beta(m\lambda, X[m]) \lambda + \sum_{m=0}^{n-1} \gamma(m\lambda, X[m]) d_w[m]$$
        :param X0: the initial value of the X, ie $X[0]$
        '''

        

    def getPaths(M:int, X0:float, mu:float, sigma:float):
        '''
        Get sample path 
        '''
        mu = 1
        n = 50
        dt = 0.1
        x0 = 100
        np.random.seed(12)

        sigma = np.arange(0.8, 2, 0.2)

        x = np.exp(
            (mu - sigma ** 2 / 2) * dt
            + sigma * np.random.normal(0, np.sqrt(dt), size=(len(sigma), n)).T
        )
        x = np.vstack([np.ones(len(sigma)), x])
        x = x0 * x.cumprod(axis=0)

        plt.plot(x)
        plt.legend(np.round(sigma, 2))
        plt.xlabel("$t$")
        plt.ylabel("$x$")
        plt.title(
            "Realizations of Geometric Brownian Motion with different variances\n $\mu=1$"
        )
        plt.show()