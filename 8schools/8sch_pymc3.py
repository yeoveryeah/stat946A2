import pymc3 as pm 
import numpy as np
import math
import matplotlib.pyplot as plt

y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
N = 8

# Create a new Model object
sch_mod = pm.Model()

with sch_mod:
    
  # hyperparameter:
  lam = pm.Uniform('lambda', lower= -1e4, upper= 1e4)
  tau2 = pm.InverseGamma('tau2', alpha= 1.0 , beta= 1.0)
  tau = pm.Deterministic('tau', pm.math.sqrt(tau2))
  
  # prior
  mu = pm.Normal('mu', lam, sigma = tau, shape = N)
  
  # Likelihood
  y = pm.Normal('y', mu, sigma=sigma, observed=y)
  

# Model fitting: inference
# PyMC3 automatically initializes NUTS to reasonable values based on the
#   variance of the samples obtained during a tuning phase.
with sch_mod:
    trace_sch = pm.sample(draws=1000)

# Posterior analysis
pm.traceplot(trace_sch);
