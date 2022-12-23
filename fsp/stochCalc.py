import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SDE:
    def __init__(self, X0 : float, lambda_t : float, N : int, Beta, Gamma) -> None:
        '''
        Define an SDE Object, which simulates an SDE of the form: 

        $$X[n] = X[0] + \sum_{m=0}^{n-1} \beta(m\lambda, X[m]) \lambda + \sum_{m=0}^{n-1} \gamma(m\lambda, X[m]) d_w[m]$$
        :param X0: the initial value of the X, ie $X[0]$
        :param lambda_t: the lambda in the equation above (the unit of discretized time)
        :param N: the length of time simulated
        :param Beta: a function that takes in $m*\lambda$, and $X[m]$, and returns a floating point value. 
        :param Gamma: same as above.
        '''

        self.X0 = X0
        self.lambda_t = lambda_t
        self.N = N
        self.Beta = Beta 
        self.Gamma = Gamma
        self.badPathCounter = 0

    def getValidPath(self):
        
        W = np.random.standard_normal(size=self.N+1)
        W = np.cumsum(W)*np.sqrt(self.lambda_t)
        X = np.zeros(self.N)

        drift = 0
        driven = 0

        X[0] = self.X0
        for n in range(1, self.N):
            drift =+ self.Beta(X[n-1]) * self.lambda_t
            driven =+ self.Gamma(X[n-1]) *(W[n]-W[n-1])
            X[n] = X[n-1] + drift + driven

            if(X[n] < 0):
                self.badPathCounter += 1
                Warning("A path with Negative X0 was generated, retrying. BAD PATH #", self.badPathCounter)
                return self.getValidPath()

        return X

    def getValidPaths(self, M=1000):
        for _ in range(M):
            yield self.getValidPath()