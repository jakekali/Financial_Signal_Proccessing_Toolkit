import numpy as np 
from simulation import *

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

    def simulatePathDependent(self, S0, T, M, p, max=False, maxVal=0):
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
        
        if(not max):

            coin_flips = self.rng.choice([0, 1], size=(T,M), p=[(1-p),p])

            SN = self.stockPrice(len(coin_flips)-sum(coin_flips), sum(coin_flips))
            V0 = self.V(SN)

            return SN/(1+self.risk_free_rate)**T, V0/(1+self.risk_free_rate)**T

        else:
            coin_flips = self.rng.choice([0, 1], size=(T,M), p=[(1-p),p])
            S = np.zeros((T+1, M))
            MAX = np.zeros((T+1, M)).fill(maxVal)

            S[0] = S0
            for t in range(1, T+1):
                S[t] = S[t-1]*self.up_factor**coin_flips[t-1]*self.down_factor**(1-coin_flips[t-1])
                MAX[t] = np.maximum(MAX[t-1], S[t] * (1+self.risk_free_rate)**(T-t))

            return S[-1]/(1+self.risk_free_rate)**T, (MAX[-1] - S[-1])/(1+self.risk_free_rate)**T
        

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
        
    
