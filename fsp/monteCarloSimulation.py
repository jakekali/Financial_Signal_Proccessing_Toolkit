import numpy as np 
from simulation import simulation

class MonteCarlo(simulation):
    '''
    A Monte Carlo Simulation
    '''
    def setRNG(self, rng):
        '''
        Set the random number generator
        :param rng: The random number generator
        '''
        if(rng is None):
            self.rng = np.random.RandomState()
        else:
            self.rng = rng


    def isArbitrage(self):
        '''
        Check if the simulation is an arbitrage
        :return: True if the simulation is an arbitrage, False otherwise
        '''
        return (1+self.r) > self.u or (1+self.r) < self.d

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
        if(T < 0 or type(T) != int):
            raise ValueError('The number of periods must must be positive, integer')
        if(M < 0 or type(M) != int):
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
            return S0* self.V_N(coin_flips_pos)
        else:
            raise ValueError('Path dependent is True, but the value function is path independent. Try using the simulatePathDependent() function')
        
    
