import numpy as np

class ARCH:
    def __init__(self, p):
        self.p = p
        self.coefs = np.zeros(p)
        self.variance = 0.0
        
    def fit(self, data, max_iters=1000, tol=1e-6):
        # Initialize the variance and coefficients
        self.variance = data.var()
        for i in range(self.p):
            self.coefs[i] = 0.1 #initial value might be very bad?
            
        # Run the estimation loop
        for t in range(max_iters):
            old_variance = self.variance
            errors = data - np.dot(self.coefs, data[t-self.p:t][::-1])
            variance = np.mean(errors**2)
            self.variance = (1 - self.coefs.sum()) * variance + np.dot(self.coefs, errors[1:]**2)
            self.coefs = variance * errors[1:]**2 / self.variance
            
            # Check for convergence
            if np.abs(old_variance - self.variance) < tol:
                break
                
    def predict(self, data):
        # Predict the volatility of the next time step
        errors = data - np.dot(self.coefs, data[-self.p:][::-1])
        return (1 - self.coefs.sum()) * self.variance + np.dot(self.coefs, errors**2)