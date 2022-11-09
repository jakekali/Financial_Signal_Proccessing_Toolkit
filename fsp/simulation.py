class simulation:
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

    def riskNeutralProbability(self):
        '''
        Calculate the risk neutral probability

        :return: The risk neutral probability of up
        '''
        return ((1+self.risk_free_rate) - self.down_factor) / (self.up_factor - self.down_factor)

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
