import pymc3 as pm 
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 20
true_lam = 3 # true population mean
true_tau2 = 1.5 # true between-cluster variance
true_sig2 = 1.2 # true within-cluster variance 
true_mu = true_lam + np.sqrt(true_tau2)*np.random.normal(size = N) # true cluster means
y = true_mu + np.sqrt(true_sig2)*np.random.normal(size = N) # observed data input for models

# Create a new Model object
hnm_mod = pm.Model()

with hnm_mod:
    
  # hyperparameter:
  lam = pm.Uniform('lambda', lower= -1e4, upper= 1e4)
  tau2 = pm.InverseGamma('tau2', alpha= 1.0 , beta= 1.0)

  # prior
  mu_raw = pm.Normal('mu_raw', mu=0,sigma=1)
  mu = pm.Deterministic('mu', lam + np.sqrt(tau2)*mu_raw)
  sigma2 = pm.InverseGamma('sigma2', alpha=1.0, beta=1.0)
  # Likelihood
  y = pm.Normal('y', mu, sigma= np.sqrt(sigma2), observed=y)
  

# Model fitting: inference
# PyMC3 automatically initializes NUTS to reasonable values based on the
#   variance of the samples obtained during a tuning phase.
with hnm_mod:
    trace_hnm = pm.sample(draws=10000, chains=4)

# Posterior analysis
pm.traceplot(trace_hnm)
