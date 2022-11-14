import numpy as np
import scipy.stats as ss
from simulation import *

class bapm_exact(simulation):

    replicatingDeltas = []

    def pathIndependentExpectedValue(self, T, p):
        '''
        Compute the expected value of the option using the path independent method
        :param T: The number of periods (the number of coin flips)
        :param p: The probability of up
        :return: A tuple wit the expected value of the option, and the expected value of the underlying stock
        '''
        
        if(T < 0 or type(T) != int):
            raise ValueError('The number of periods must must be positive, integer')

        tot = 0
        eVals = 0 # The expected value of the option
        eStocks = 0 # The expected value of the stock
        histogram = ss.binom(T, p)
        for k in range(T+1):           
            tot += histogram.pmf(k)
            eStocks += histogram.pmf(k) * self.stockPrice(T-k, k)
            eVals += self.V(self.stockPrice(T-k, k)) * histogram.pmf(k)

        #check that the probabilities sum to 1
        assert (round(tot,7) == 1)

        return (eVals / (1+self.risk_free_rate)**T)[0], eStocks / (1+self.risk_free_rate)**T

    def singleStepReplicatingPortfolio(self, VH, VL):
        '''
        For any option, there is a replicating portfolio that has the same value as the option.

        :param VH: The value of the option if heads
        :param VL: The value of the option if tails

        :return: A tuple containing the fair price of the option at the initial time, and the number of shares of the underlying stock to buy
        '''
       
        

        A = np.array([[(1+self.risk_free_rate), (self.up_factor-(1+self.risk_free_rate))*self.S0], 
                      [(1+self.risk_free_rate), (self.down_factor-(1+self.risk_free_rate))*self.S0]],
                     dtype=np.double)

        b = np.array([VH, VL], dtype=np.double)

        return np.linalg.solve(A, b)

    def recursiveReplicatingPortfolio(self, maxPathLength=3, path=[]):
        '''
        For any option, there is a replicating portfolio that has the same value as the option.

        :param maxPathLength: The maximum path length to consider
        :param path: The path to consider

        :return: A tuple containing the fair price of the option at the initial time, and the number of shares of the underlying stock to buy
        '''
        self.replicatingDeltas = []
        self.replicatingDeltas = [None] * ((2**(maxPathLength))-1)
        c = self.__recursiveReplicatingPortfolio(maxPathLength, path)
        return c[0], self.replicatingDeltas

    def __recursiveReplicatingPortfolio(self, maxPathLength=3, path=[]):
        '''
        For any option, there is a replicating portfolio that has the same value as the option.

        :param maxPathLength: The maximum path length to consider
        :param path: The path to consider

        :return: A tuple containing the fair price of the option at the initial time, and the number of shares of the underlying stock to buy
        '''

        if(len(path) != maxPathLength):
            pathT = path + [0]
            pathH = path + [1] 
        else: 
            return self.V(self.stockPrice(len(path) - sum(path), sum(path))), 1

        VL = self.__recursiveReplicatingPortfolio(maxPathLength, pathT)
        VH = self.__recursiveReplicatingPortfolio(maxPathLength, pathH)

        print("VH ", VH, "VL ", VL)

        V0, d = self.singleStepReplicatingPortfolio(VH[0], VL[0])

        assert(self.replicatingDeltas[self.traverse(maxPathLength=maxPathLength, path=path)] == None)
        self.replicatingDeltas[self.traverse(maxPathLength=maxPathLength, path=path)] = np.double(d)

        return np.double(V0), np.double(d)
    

        

    




       

    
            








    


    