# hnm_pm3.py:
import pymc3 as pm 
import numpy as np
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
import arviz as az

# 1. Data
# To insure the PyMC3 is working on the same set of data as in R, 
#   rpy2, an inference to R in Python, is implementned:
data = robjects.r("""
set.seed(42) 
N <- 20 
true_lam <- 3
true_tau2 <- 1.5 
true_sig2 <- 1.2
true_mu <-true_lam + sqrt(true_tau2)*rnorm(N) 
y <- true_mu + sqrt(true_sig2)*rnorm(N)
""")
#  The true parameters are given as: 
N = 20
true_lam = 3
true_tau2 = 1.5 
true_sig2 = 1.2
y = data
nsamples = 1e4

# 2. Model fitting:
# Building the model:
#   pm.Model() creates a new Model object, the subsequent specification of the model components 
# is performed inside a `with` statement until the indented block ends.
with pm.Model() as hnm_mod:  

  # hyper-priors:
  lam_tilde = pm.Normal('lambda_tilde', mu=0, sigma=1)
  lam = pm.Deterministic('lambda', 10 * lam_tilde)
  tau_sq = pm.InverseGamma('tau_sq', alpha= 1.0 , beta= 1.0, transform=None)
  # prior:
  sigma_sq = pm.InverseGamma('sigma_sq', alpha=1.0, beta=1.0, transform=None)
  # Likelihood:
  y_sd = pm.Deterministic('y_sd', np.sqrt(sigma_sq + tau_sq))
  L = pm.Normal('Likelihood', lam, sigma= y_sd, observed=y)

# Model fitting: 
#   Different from Stan, the draws and warm-up draws are specified separately.
with hnm_mod:
  trace_hnm = pm.sample(draws= nsamples/2, tune=nsamples/2, target_accept=0.99, chains=4, cores=1)

# 3. Traceplot:
# `pm.traceplot` will generate trace plot for selected parameters as well
with hnm_mod:
  az.plot_trace(trace_hnm, var_names = ["lambda", "tau_sq", "sigma_sq"], legend=True)
  plt.show()

# 4. Posterior plot:
# `pm.plot_posterior() will generate posterior plot as well:
with hnm_mod:
  az.plot_posterior(trace_hnm, var_names = ["lambda", "tau_sq", "sigma_sq"], kind='hist', point_estimate='mean', 
    ref_val=[true_lam, true_tau2, true_sig2], hdi_prob='hide', credible_interval=None)
  plt.show()