import numpy as np 

class MonteCarlo:
    '''
    A Monte Carlo Simulation
    '''

    def __init__(self, r, d=0.5, u=1.5, pathDependent=None, V_N=None, arbitrage='raise', rng=None):
        '''
        Initialize the simulation

        :param r: The risk free rate
        :param d: The down factor, if pathDependent is not None then this is ignored
        :param u: The up factor, if pathDependent is not None then this is ignored

        :param pathDependent: a flag that indicates if the value function is path dependent
        if pathDependent is None, then the value function is derived from the up and down factors,
        by the formula $V_{n+1} = ( u \\times H \\plus (self.d) \\times (1 - H))$, where $H$ is a random variable
        that is 1 with probability $p$ and 0 with probability $1 - p$. 

        if pathDependent is True, then the value function V_N will be used. The function V_N must take in a (T, N) array 
        of simulated coin flips and return a (1, N) array of final values.

        if pathDependent is False, then the value function must take in a (1, N) array indicating the number of heads
        and return a (1, N) array of final values.

        :param V_N: The value function as described above, in the pathDependent parameter. If pathDependent is None, then this is ignored.
        :param arbitrage: The behavior when the simulation is an arbitrage. If 'raise', then an exception is raised. If 'ignore', then the simulation is run anyway.
        :param rng: The random number generator (np.random.RandomState)
        '''
        if(r <= 0):
                raise ValueError('r must be greater than 0')

        self.r = r
        self.pathDependent = pathDependent

        if(pathDependent is None):
            if(d == u):
                raise ValueError('d and u must be different')

            self.d = d
            self.u = u

            if(self.isArbitrage() and arbitrage == 'raise'):
                raise ValueError('Arbitrage detected')

        else:
            self.V_N = V_N

        if(rng is None):
            self.rng = np.random.RandomState()
        else:
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

        :return: A (T, N) array of simulated stock prices, at each time step, at the value at each time step
        '''

        if(S0 <= 0):
            raise ValueError('S0 must be positive')
        if(T < 0 and type(T) != int):
            raise ValueError('The number of periods must must be positive, integer')
        if(M < 0 and type(M) != int):
            raise ValueError('The number of simulations must be positive, integer')

        S = np.zeros((T, M))
        coin_flips = self.rng.choice([0, 1], size=(T,M), p=[(1-p),p])

        if(self.pathDependent is None):
            S[0, :] = S0
            for t in range(1, T):
                S[t, :] = S[t-1, :] * ((self.u) * coin_flips[t, :] + (self.d) * (1 - coin_flips[t, :]))

            return S
        if(self.pathDependent):
            return self.V_N(coin_flips)
        else:
            Warning('Path dependent is False, but the value function is path dependent. Using path dependent value function')
            return self.V_N(np.sum(coin_flips, axis=0))

        

    def simulatePathIndependent(self, S0, T, M, p):
        '''
        Simulate the stock price at the end of a Monte Carlo simulation

        :param S0: The initial stock price
        :param T: The number of periods
        :param M: The number of simulations
        :param p: The probability of the stock going up
        :return S_T: An array of simulated stock prices at the end of the simulation, at value at final time step
        '''

        if(S0 <= 0):
            raise ValueError('S0 must be positive')
        if(T < 0 and type(T) != int):
            raise ValueError('The number of periods must must be positive, integer')
        if(M < 0 and type(M) != int):
            raise ValueError('The number of simulations must be positive, integer')

        coin_flips_pos = self.rng.binomial(T, p, M)
        if self.pathDependent is None:
            return S0 * (self.u ** coin_flips_pos) * (self.d ** (T - coin_flips_pos))
        elif(self.pathDependent is not None and not self.pathDependent):
            return self.V_N(coin_flips_pos)
        else:
            raise ValueError('Path dependent is True, but the value function is path independent. Try using the simulatePathDependent() function')
        

class BAPM:

    @staticmethod
    def riskNeutralProbability(r, d, u):
        '''
        Calculate the risk neutral probability

        :param r: The risk free rate
        :param d: The down factor
        :param u: The up factor

        :return: The risk neutral probability of up
        '''
        return ((1+r) - d) / (u - d)

    @staticmethod
    def stepWealthEquation(X0, S0, SN, dn, r):
        '''
        Calculate the wealth equation

        :param X0: The initial value of the portfolio 
        :param S0: The initial stock price
        :param SN: The stock price at the next step
        :param dn: the number of shares of each stock in the portfolio
        :param r: The risk free rate

        :return: The value of the portfolio at the next step
        '''

        return np.dot(dn, SN) + (1 + r) * (X0 - np.dot(dn, S0))



