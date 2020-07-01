# hnm_pymc3.py:
import pymc3 as pm 
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os

# 1. Data
# To insure the PyMC3 is working on the same set of data as in R, 
#   rpy2, an inference to R in Python, is implementned:
#  The true parameters are given as: 
#  The true parameters are given as: 
N = 20
true_lam = 3
true_tau2 = 1.5 
true_sig2 = 1.2
nsamples = 1e4
# Load the observations:
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(THIS_FOLDER, 'data.txt')
d = open(data_file)
y = np.array([float(line.strip('\n')) for line in d.readlines()])

# 2. Model fitting:
# Building the model:
#   pm.Model() creates a new Model object, the subsequent specification of the model components 
# is performed inside a `with` statement until the indented block ends.
with pm.Model() as hnm_mod:  

  # hyper-priors:
  lam_tilde = pm.Normal('lambda_tilde', mu=0, sigma=1)
  lam = pm.Deterministic('lambda', 10 * lam_tilde)
  tau_sq = pm.InverseGamma('tau_sq', alpha= 1.0 , beta= 1.0)
  # prior:
  sigma_sq = pm.InverseGamma('sigma_sq', alpha=1.0, beta=1.0)
  # Likelihood:
  y_sd = pm.Deterministic('y_sd', np.sqrt(sigma_sq + tau_sq))
  L = pm.Normal('Likelihood', lam, sigma= y_sd, observed=y)

# Check the correctness of the model
hnm_mod.vars # return list of variables 
# log-posterior
logp_nojac = hnm_mod.logp_nojac 
print(logp_nojac({'lambda_tilda': 1, 'tau_sq_log__': np.log(5), 'sigma_sq_log__': np.log(4)}))
# log-posterior including log of jacobian determinants of transformed variables
logp = hnm_mod.logp 
print(logp({'lambda_tilda': 1, 'tau_sq_log__': np.log(5), 'sigma_sq_log__': np.log(4)}))
np.log(5)+np.log(4) # Jacobian terms

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
    ref_val=[true_lam, true_tau2, true_sig2], hdi_prob=0.90)
  plt.show()