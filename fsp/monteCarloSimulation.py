import numpy as np 

class MonteCarlo:
    '''
    A Monte Carlo Simulation
    '''

    def __init__(self, r, d, u, rng):
        '''
        Initialize the simulation

        :param r: The risk free rate
        :param d: The down factor
        :param u: The up factor
        :param rng: The random number generator (np.random.RandomState)
        '''

        if(r <= 0 or r >= 1):
            raise ValueError('r must be in (0, 1)')
        if(d == u):
            raise ValueError('d and u must be different')

        self.r = r + 1
        self.d = d
        self.u = u

        if(self.isArbitrage()):
            raise ValueError('Arbitrage detected')

        self.rng = rng


    def isArbitrage(self):
        '''
        Check if the simulation is an arbitrage
        :return: True if the simulation is an arbitrage, False otherwise
        '''
        return self.r > self.u or self.r < self.d

    def simulatePathDependent(self, S0, T, M, p):
        '''
        Simulate the stock price

        :param S0: The initial stock price
        :param T: The number of periods
        :param M: The number of simulations
        :param p: The probability of the stock going up
        :return: A (T, N) array of simulated stock prices, atr each time step
        '''

        if(S0 <= 0):
            raise ValueError('S0 must be positive')
        if(T < 0 and type(T) != int):
            raise ValueError('The number of periods must must be positive, integer')
        if(M < 0 and type(M) != int):
            raise ValueError('The number of simulations must be positive, integer')

        S = np.zeros((T, M))
        coin_flips = self.rng.choice([0, 1], size=(T,M), p=[(1-p),p])

        S[0, :] = S0
        for t in range(1, T):
            S[t, :] = S[t-1, :] * ((self.u) * coin_flips[t, :] + (self.d) * (1 - coin_flips[t, :]))

        return S

    def simulatePathIndependent(self, S0, T, M, p):
        '''
        Simulate the stock price at the end of a Monte Carlo simulation

        :param S0: The initial stock price
        :param T: The number of periods
        :param M: The number of simulations
        :param p: The probability of the stock going up
        :return S_T: An array of simulated stock prices at the end of the simulation
        '''

        if(S0 <= 0):
            raise ValueError('S0 must be positive')
        if(T < 0 and type(T) != int):
            raise ValueError('The number of periods must must be positive, integer')
        if(M < 0 and type(M) != int):
            raise ValueError('The number of simulations must be positive, integer')

        coin_flips_pos = self.rng.binomial(T, p, M)
        coin_flips_neg = T - coin_flips_pos

        return S0 * (self.u ** coin_flips_pos) * (self.d ** coin_flips_neg)

        



