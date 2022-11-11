import sys 
sys.path.append('../fsp')
import monteCarloSimulation as mcs
import bapm_exact as be
import numpy as np
import math

# 1C. Generate a MultiStep replicating portfolio
# Test with example from notes.

S0 = 100
K = 5
u = 1.5
d = 0.95
r = 0.45

exp = be.bapm_exact(u, d, r, S0)
exp.setOption(True, True, K, 3)
VH = exp.V(S0*u)
VL = exp.V(S0*d)
V0, Deltas = exp.recursiveReplicatingPortfolio(3)
print(r"$ E[V_N] = " + str( V0) + r"$ " + r", $ \Delta = " + str( Deltas) + r"$")
# The Deltas are stored in a binary tree, so the first element is the root, 
# the second and third are the children of the root, 
# the fourth and fifth are the children of the second, and so on.

# We can test the above by computing the expected value of the option using the replicating portfolio
c = exp.pathIndependentExpectedValue(3, exp.riskNeutralProbability())
print(c)


print(exp.generateAllPaths(3))
# We can also test, by verifying that the value of the option is the same as the value of the replicating portfolio, for all possible paths.
exp.verifyReplicatingPortfolio(3)

