import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


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
        self.badPathsSample = []

    def getValidPath(self, greaterThanZero=True, stopBelowZero=True):
        
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

            if((X[n] < 0 and stopBelowZero)):
                self.badPathCounter += 1
                Warning("A path with Negative X0 was generated, retrying. BAD PATH #", self.badPathCounter)
                self.badPathsSample.append(X)
                if greaterThanZero:
                    return self.getValidPath()
                return X
        return X

    def getValidPaths(self, M=1000, greaterThanZero=True, stopBelowZero=True):
        self.badPathCounter = 0
        for _ in range(M):
            path = self.getValidPath(greaterThanZero=greaterThanZero,stopBelowZero=stopBelowZero)
            yield path

class SDE_Efficient:
    def __init__(self, X0 : float, lambda_t : float, N : int, M:int):
        self.X = np.zeros((M, N))
        self.X[:,0] = np.ones((1,M)) * X0
        self.X0 = X0
        self.M = M
        self.N = N
        self.lambda_t = lambda_t
        
    def walk(self, Beta, Gamma, replace0=False):
        W = np.random.standard_normal(size=(self.M, self.N+1))
        W = np.cumsum(W,axis=1)*np.sqrt(self.lambda_t)*2
        
        if(not replace0):
            for time in range(1,self.N):  
                x= self.X[:,time-1] + Beta(self.X[:,time-1]) * self.lambda_t + Gamma(self.X[:,time-1]) * (W[:,time]-W[:,time-1])
                self.X[:,time] = x
            self.X = self.X[~np.isnan(self.X).any(axis=1), :]

        else:
            # repeat until no nan values are present, and remove rows with nans are recalculate them
            for time in range(1,self.N):  
                    x= self.X[:,time-1] + Beta(self.X[:,time-1]) * self.lambda_t + Gamma(self.X[:,time-1]) * (W[:,time]-W[:,time-1])
                    self.X[:,time] = x
                    
            self.BadPaths = self.X[np.isnan(self.X).any(axis=1), :]
            self.X = self.X[~np.isnan(self.X).any(axis=1), :]
            
            attempt_num = []
            attempt_num.append(self.X.shape[0])
            while(self.X.shape[0] != self.M):
                W = np.random.standard_normal(size=(self.M - self.X.shape[0], self.N+1))
                W = np.cumsum(W,axis=1)*np.sqrt(self.lambda_t)*2

                x_temp = np.zeros((self.M - self.X.shape[0], self.N))
                x_temp[:,0] = np.ones((1,self.M - self.X.shape[0])) * self.X0

                for time in range(1,self.N):
                    x = x_temp[:,time-1] + Beta(x_temp[:,time-1]) * self.lambda_t + Gamma(x_temp[:,time-1]) * (W[:,time]-W[:,time-1])
                    x_temp[:,time] = x

                x_temp = x_temp[~np.isnan(x_temp).any(axis=1), :]
                self.X = np.vstack([self.X, x_temp])
                attempt_num.append(self.X.shape[0])

            return attempt_num
                #append x_temp to X
                
                    




