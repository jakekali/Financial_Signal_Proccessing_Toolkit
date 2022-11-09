import numpy as np
import scipy.stats as ss
from simulation import simulation

class bapm_exact(simulation):

    replicatingDeltas = []

    def pathIndependentExpectedValue(self, T, p):
        '''
        Compute the expected value of the option using the path independent method
        :param T: The number of periods (the number of coin flips)
        :param p: The probability of up
        :return: The expected value of the option
        '''
        
        if(T < 0 or type(T) != int):
            raise ValueError('The number of periods must must be positive, integer')

        tot = 0
        evals = 0 # The expected value of the option
        Estocks = 0 # The expected value of the stock
        histogram = ss.binom(T, p)
        for k in range(T+1):           
            tot += histogram.pmf(k)
            Estocks += histogram.pmf(k) * self.stockPrice(T-k, k)
            evals += self.V(self.stockPrice(T-k, k)) * histogram.pmf(k)

        assert (round(tot,6) == 1)

        return evals / (1+self.risk_free_rate)**T, Estocks / (1+self.risk_free_rate)**T

    def singleStepReplicatingPortfolio(self, VH, VL):
        '''
        For any option, there is a replicating portfolio that has the same value as the option.

        :param VH: The value of the option if heads
        :param VL: The value of the option if tails

        :return: A tuple containing the fair price of the option at the initial time, and the number of shares of the underlying stock to buy
        '''
       
        

        A = np.array([[(1+self.risk_free_rate), (self.up_factor-(1+self.risk_free_rate))*self.S0], 
                    [(1+self.risk_free_rate), (self.down_factor-(1+self.risk_free_rate))*self.S0]])

        b = np.array([VH, VL])

        return np.linalg.solve(A, b)

    def recursiveReplicatingPortfolio(self, maxPathLength=3, path=[]):
        '''
        For any option, there is a replicating portfolio that has the same value as the option.

        :param maxPathLength: The maximum path length to consider
        :param path: The path to consider

        :return: A tuple containing the fair price of the option at the initial time, and the number of shares of the underlying stock to buy
        '''
        self.replicatingDeltas = []
        self.replicatingDeltas = [None] * 2**(maxPathLength)
        
        return self.__recursiveReplicatingPortfolio(maxPathLength, path)[0], self.replicatingDeltas

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

        

        V0, d = self.singleStepReplicatingPortfolio(VH[0], VL[0])


        assert(self.replicatingDeltas[self.traverse(maxPathLength=maxPathLength, path=path)] == None)
        self.replicatingDeltas[self.traverse(maxPathLength=maxPathLength, path=path)] = d

        return V0, d 
    

    def verifyReplicatingPortfolio(self, maxPathLength=3, deltas=None):
        '''
        Verify the replicating portfolio

        :param maxPathLength: The maximum path length to consider
        :param deltas: The deltas of the replicating portfolio, if none used the deltas calculated by the recursiveReplicatingPortfolio function
        '''
        experiment = self.recursiveReplicatingPortfolio(maxPathLength, path=[])
        V0 = experiment[0]

        if(deltas == None):
            deltas = experiment[1]
        if(deltas == None):
            raise ValueError("No deltas to verify")

        print(deltas)

        # Generate an array of all the possible paths
        paths = self.generateAllPaths(maxPathLength)
        for pather in paths:
            X = V0
            for step in enumerate(pather):

                path_0 = pather[:step[0]]
                path_1 = pather[:(step[0] + 1)]
                delta = deltas[self.traverse(maxPathLength=maxPathLength, path=path_0)]
                print("delta: ", delta)
                print("path_0: ", path_0)
                print("self.traverse(): ", self.traverse(maxPathLength=maxPathLength, path=path_0))

                if(delta == 0):
                    continue

                money_market = X - delta * self.stockPrice(len(path_0) - sum(path_0), sum(path_0))
                stocks = delta * self.stockPrice(len(path_1) - sum(path_1), sum(path_1))
                
                X = money_market*(1 + self.risk_free_rate) + stocks
            
            print("Path: ", pather, "X: ", X, "V: ", self.V(self.stockPrice(len(pather) - sum(pather), sum(pather))))
            assert(X == self.V(self.stockPrice(len(pather) - sum(pather), sum(pather))))

    




       

    
            








    


    