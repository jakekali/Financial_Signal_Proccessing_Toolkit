import numpy as np
import scipy.stats as ss

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
    def singleStepReplicatingPortfolio(S0, u, d, r, VH, VL):
        '''
        For any option, there is a replicating portfolio that has the same value as the option.

        :param S0: The initial value of the underlying stock
        :param u: The factor the stock will increase if heads
        :param d: The factor the stock will decrease if tails
        :param r: The risk free rate
        :param VH: The value of the option if heads
        :param VL: The value of the option if tails

        :return: A tuple containing the fair price of the option at the initial time, and the number of shares of the underlying stock to buy
        '''

        if(VH == VL and VH == 0):
            return 0, 0

        A = np.array([[(1+r), (u-(1+r))*S0], 
                    [(1+r), (d-(1+r))*S0]])

        b = np.array([VH, VL])

        return np.linalg.solve(A, b)

    @staticmethod
    def optionValueFromRiskNeutralMeasure(S0, u, d, r, VH, VL):
        '''
        For any option, there is a replicating portfolio that has the same value as the option.
        This function use the risk neutral measure to calculate the value of the option.
        This is more computationally efficient than the singleStepReplicatingPortfolio function, however it is
        more susceptible to numerical errors, and does not provide the number of shares to buy.

        :param S0: The initial value of the underlying stock
        :param u: The factor the stock will increase if heads
        :param d: The factor the stock will decrease if tails
        :param r: The risk free rate
        :param VH: The value of the option if heads
        :param VL: The value of the option if tails

        :return: The fair price of the option at the initial time
        '''

        return 1/(1+r) * (BAPM.riskNeutralProbability(r, d, u) * VH + (1-BAPM.riskNeutralProbability(r, d, u)) * VL)

    @staticmethod
    def multiStepReplicatingPortfolio(S0, u, d, r, M):
        '''
        For any option, there is a replicating portfolio that has the same value as the option.

        :param S0: The initial value of the underlying stock
        :param u: The factor the stock will increase if heads
        :param d: The factor the stock will decrease if tails
        :param r: The risk free rate
        :param M: The number of periods
        :param VH: The value of the option if heads
        :param VL: The value of the option if tails

        :return: A list of tuples containing the the fair price of the option at the initial time, and the number of shares of the underlying stock to buy
        '''

        return BAPM._recursiveReplicatingPortfolio(u, d, S0, r, path=[], maxPathLength=M)

    @staticmethod
    def _recursiveReplicatingPortfolio(u, d, S0, r, path=[], maxPathLength=3):
        '''
        Helper function for the multiStepReplicatingPortfolio function

        :param u: The factor the stock will increase if heads
        :param d: The factor the stock will decrease if tails
        :param S0: The initial value of the underlying stock
        :param path: The path of the stock price

        :return: A list of tuples containing the the fair price of the option at the initial time, and the number of shares of the underlying stock to buy
        '''

        if(len(path) != maxPathLength):
            pathT = path + [0]
            pathH = path + [1]
        else:
            print("\t"*len(path), path, " : ", S0 * BAPM.valueFunctionP(path, d, u))
            return max(S0 * BAPM.valueFunctionP(path, d, u) - 5, 0), 0 


        VH, upf = BAPM._recursiveReplicatingPortfolio(u, d, S0, r, path=pathH, maxPathLength=maxPathLength)
        VL, dof = BAPM._recursiveReplicatingPortfolio(u, d, S0, r, path=pathT, maxPathLength=maxPathLength)


        print("\t"*len(path), "path: ", path, " : ", BAPM.optionValueFromRiskNeutralMeasure(S0, u, d, r, VH, VL), " : ", upf, " : ", dof)
        print("\t"*len(path), "path: ", path, " : ", BAPM.valueFunctionP(path, d, u), u, d, r, VH, VL)
        V0, delta = BAPM.singleStepReplicatingPortfolio(S0*BAPM.valueFunctionP(path, d, u), u, d, r, VH, VL)
        print("\t"*len(path), "path: ",path, " V0: ", V0, " D: ", delta)
        return V0, delta


    @staticmethod
    def valueFunctionP(path, d, u):
        '''
        Calculate the value function for a given path

        :param path: The path to calculate the value function for
        :param d: The down factor
        :param u: The up factor

        :return: The value function
        '''
        factor = 1
        for i in path:
            if i == 0:
                factor = factor*d
            elif i == 1:
                factor = factor*u

        return factor

    @staticmethod
    def valueFunction(H, D, d, u):
        '''
        The value function for a path independent option

        :param H: The number of heads
        :param D: The number of tails
        :param d: The down factor
        :param u: The up factor

        :return: The value of the option
        '''

        return (u**H) * (d**D)


    

class probabilistic_analysis:

    def __init__(self, r, d=0.5, u=1.5, pathDependent=None, V_N=None, arbitrage='raise'):
        '''

        :param r: The risk free rate
        :param d: The down factor, if pathDependent is not None then this is ignored
        :param u: The up factor, if pathDependent is not None then this is ignored

        :param pathDependent: a flag that indicates if the value function is path dependent
        if pathDependent is None, then the value function is derived from the up and down factors,
        by the formula $V_{n+1} = ( u \\times H + (self.d) \\times (1 - H))$, where $H$ is a random variable
        that is 1 with probability $p$ and 0 with probability $1 - p$. 

        if pathDependent is True, then the value function V_N will be used. The function V_N must take in a (T, N) array 
        of simulated coin flips and return a (1, N) array of final values.

        if pathDependent is False, then the value function must take in a (1, N) array indicating the number of heads
        and return a (1, N) array of final values.

        :param V_N: The value function as described above, in the pathDependent parameter. If pathDependent is None, then this is ignored.
        :param arbitrage: The behavior when the simulation is an arbitrage. If 'raise', then an exception is raised. If 'ignore', then the simulation is run anyway.
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



    def isArbitrage(self):
        '''
        Check if the simulation is an arbitrage
        :return: True if the simulation is an arbitrage, False otherwise
        '''
        return (1 + self.r) > self.u or (1+self.r) < self.d

    def pathIndependentExpectedValue(self, S0, T, p):
        '''
        Calculate the expected value of the path independent simulation

        :param S0: The initial stock price
        :param T: The number of steps
        :param p: The probability of heads

        :return: The expected value of the path independent simulation
        '''
        if(S0 <= 0):
            raise ValueError('S0 must be positive')
        if(T < 0 or type(T) != int):
            raise ValueError('The number of periods must must be positive, integer')

        K = 5
        SN = 0
        hh = ss.binom(T, p)
        for k in range(T):

            if self.pathDependent is None:
                SN += max(0, (S0 * (self.u ** k) * (self.d ** (T - k))) - K) * hh.pmf(k)

            elif(self.pathDependent is not None and not self.pathDependent):
                SN += S0* self.V_N(k) * hh.pmf(k)

            else:
                raise ValueError('Invalid pathDependent parameter')

        return SN
