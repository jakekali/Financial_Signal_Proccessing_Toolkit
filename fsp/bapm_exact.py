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
    

    def verifyReplicatingPortfolio(self, maxPathLength=3):
        '''
        Verify the replicating portfolio

        :param maxPathLength: The maximum path length to consider
        :param deltas: The deltas of the replicating portfolio, if none used the deltas calculated by the recursiveReplicatingPortfolio function
        '''
        experiment = self.recursiveReplicatingPortfolio(maxPathLength, path=[])
        V0 = experiment[0]

        deltas = self.replicatingDeltas
        if(deltas == None):
            raise ValueError("No deltas to verify")


        print(deltas)

        # Generate an array of all the possible paths
        paths = self.generateAllPaths(maxPathLength)
        for pather in paths:
            X = np.longfloat(V0)
            for step in enumerate(pather):

                path_0 = pather[:step[0]]
                path_1 = pather[:(step[0] + 1)]
                delta = deltas[self.traverse(maxPathLength=maxPathLength, path=path_0)]
                print("delta: ", delta)
                print("path_0: ", path_0)
                print("self.traverse(): ", self.traverse(maxPathLength=maxPathLength, path=path_0))

                if(delta == 0):
                    continue

                money_market = X - delta * np.longfloat(self.stockPrice(len(path_0) - sum(path_0), sum(path_0)))
                stocks = delta * np.longfloat(self.stockPrice(len(path_1) - sum(path_1), sum(path_1)))
                print("money_market: ", money_market)
                print("stocks: ", stocks)
                
                X = money_market*np.longfloat(1 + self.risk_free_rate) + stocks
            
            print("Path: ", pather, "X: ", X, "V: ", self.V(self.stockPrice(len(pather) - sum(pather), sum(pather))))
            assert(X == np.longfloat(self.V(self.stockPrice(len(pather) - sum(pather), sum(pather)))))


    def verifyReplicatingReWrite(self, V0, maxPathLength=3):
        '''
        Verify Replicating Portfolio

        :param: max path length
        '''

        paths = self.generateAllPaths(maxPathLength)
        print("p", self.replicatingDeltas)
        for path in paths: 
            indexs = self.rewriteBinaryTree(path)
            deltas = [self.replicatingDeltas[i] for i in indexs]
            S0 = self.S0
            cash = V0
            for(i, val) in enumerate(path):
                print("Start of loop")
                print("cash: ", cash)
                print("S0: ", S0)
                cash = cash - deltas[i] * S0
                print("money market: ", cash)
                print("stocks ", deltas[i] * S0)

                if(val == 0):
                    S0 = S0 * self.down_factor
                else:
                    S0 = S0 * self.up_factor
                cash = (1 + self.risk_free_rate) * cash
                cash = cash + deltas[i] * S0
                print("Path: ", path, "X: ", cash, "V: ", self.V(S0))



    def rewriteBinaryTree(self, path):
        indexes = [0]
        i = 0
        for j, step in enumerate(path):
            if(j == len(path)-1):
                break
            if(step):
                i = 2*i + 1
            else:
                i = 2*i + 2
            indexes.append(i)
        return(indexes)

        

    




       

    
            








    


    