import numpy as np
import scipy.stats as ss

class bapm_exact:

    replicatingDeltas = []

    def __init__(self, up_factor, down_factor, risk_free_rate, S0):
        '''
        :param up_factor: The up factor
        :param down_factor: The down factor
        :param risk_free_rate: The risk free rate
        '''
        self.up_factor = up_factor
        self.down_factor = down_factor
        self.risk_free_rate = risk_free_rate
        self.S0 = S0

        if(risk_free_rate <= 0):
                raise ValueError('the risk free rate must be greater than 0')

        if(self.isArbitrage()):
            raise ValueError("Arbitrage detected")

        return None

    def setOption(self, isEU, isCall, strike, maturity):
        self.isEU = isEU
        self.isCall = isCall
        self.strike = strike
        self.maturity = maturity

    def V(self, S):
        '''
        Compute the value of the option
        :param S: The stock price
        :return: The value of the option
        '''

        if(self.isEU):
            if(self.isCall):
                return self.EU_call(S)
            else:
                return self.EU_put(S)
        else:
            if(self.isCall):
                return self.call(S)
            else:
                return self.put(S)

    def EU_call(self, S1):
        return max(0, S1 - self.strike)
    
    def EU_put(self, S1):
        return max(0, self.strike - S1)

    def call(self, S1):
        return S1 - self.strike

    def put(self, S1):
        return self.strike - S1


    def isArbitrage(self):
        '''
        Check if the simulation is an arbitrage
        :return: True if the simulation is an arbitrage, False otherwise
        '''
        return (1 + self.risk_free_rate) > self.up_factor or (1+self.risk_free_rate) < self.down_factor

    def stockPrice(self, T, H):
        '''
        Compute the stock price at time T
        :param T: The number of periods (the number of coin flips)
        :param H: The number of heads
        :return: The stock price at time T
        '''
        return  self.S0 * (self.up_factor**H) * (self.down_factor**(T))

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

    def riskNeutralProbability(self):
        '''
        Calculate the risk neutral probability

        :return: The risk neutral probability of up
        '''
        return ((1+self.risk_free_rate) - self.down_factor) / (self.up_factor - self.down_factor)

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

    @staticmethod
    def traverse(maxPathLength=3, path=[]):
        '''
        Gives the Index of a particular path in a binary tree

        
        :param maxPathLength: The maximum path length to consider
        :param path: The path to consider
        :return: The index of the path in the binary tree
        '''
        if(len(path) == 0):
            return 0
        pos = 0
        for (p) in (path):
            if(p != 0 and p != 1):
                raise ValueError("Path must be a binary path")
            pos = pos * 2 + p + 1

        return pos

    

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

    @staticmethod
    def generateAllPaths(maxPathLength=3):
        '''
        Generate all the possible paths

        :param maxPathLength: The maximum path length to consider
        :return: An array of all the possible paths
        '''
        if maxPathLength == 1:
            return [[0], [1]]

        paths = [('{0:0'+ str(maxPathLength) + 'b}').format(s) for s in range(maxPathLength**2)]
        for i in range(len(paths)):
            paths[i] = [int(s) for s in paths[i]]
        return paths




       

    
            








    


    