import numpy as np

class simulation:
    
    isEU = True
    isCall = True

    def __init__(self, up_factor, down_factor, risk_free_rate, S0):
        '''
        :param up_factor: The up factor
        :param down_factor: The down factor
        :param risk_free_rate: The risk free rate
        '''
        self.up_factor = np.double(up_factor)
        self.down_factor = np.double(down_factor)
        self.risk_free_rate = np.double(risk_free_rate)
        self.S0 = np.double(S0)

        if(risk_free_rate <= 0):
                raise ValueError('the risk free rate must be greater than 0')

        if(self.isArbitrage()):
            raise ValueError("Arbitrage detected")

        return None

    def setOption(self, isEU, isCall, strike, maturity):
        '''
        Set the option type, strike and maturity
        :param isEU: True if the option is European
        :param isCall: True if the option is a call
        :param strike: The strike price
        :param maturity: The maturity
        '''

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
        S1 = np.double(S1)
        return np.maximum(np.zeros(np.size(S1 - self.strike)), S1 - self.strike)

    
    def EU_put(self, S1):
        S1 = np.double(S1)
        return np.maximum(np.zeros(np.size(self.strike - S1)), self.strike - S1)

    def call(self, S1):
        S1 = np.double(S1)
        return S1 - self.strike

    def put(self, S1):
        S1 = np.double(S1)
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
        :param T: The number of tails
        :param H: The number of heads
        :return: The stock price at time T
        '''
        T = np.double(T)
        H = np.double(H)
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

        paths = [('{0:0'+ str(maxPathLength) + 'b}').format(s) for s in range(2**maxPathLength)]
        for i in range(len(paths)):
            paths[i] = [int(s) for s in paths[i]]
        return paths

    @staticmethod
    def generateRandomPaths(maxPathLength=10, numPaths=10, p=0.5):
        '''
        Generate all the possible paths

        :param maxPathLength: The maximum path length to consider
        :param numPaths: The number of paths to generate
        :param p: The probability of heads
        :return: An array of all the possible paths
        '''
        paths = np.random.choice([0, 1], size=(numPaths, maxPathLength), p = [1-p, p])
        return paths

    def describe(self):
        '''
        Describe the simulation parameters
        :return: A string describing the simulation
        '''
        return r"u: " + str(self.up_factor) + r" d: " + str(self.down_factor) + r" \n r: " + str(self.risk_free_rate) + " S0: " + str(self.S0) + " \n strike: " + str(self.strike) + " maturity: " + str(self.maturity) + " \n"

        
