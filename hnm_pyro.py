import numpyro as po 
import numpy as np
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import rpy2.robjects as robjects

data = robjects.r("""
set.seed(42) 
N <- 20 
true_lam <- 3
true_tau2 <- 1.5 
true_sig2 <- 1.2
true_mu <-true_lam + sqrt(true_tau2)*rnorm(N) 
y <- true_mu + sqrt(true_sig2)*rnorm(N)
""")

N = 20
true_lam = 3
true_tau2 = 1.5 
true_sig2 = 1.2
y = data
nsamples = 1e4

def hnm_mod(y = None, N):
    lam_tilde = po.sample('lam_tilde', dist.Normal(0,1))
    tau2 = po.sample('tau2', dist.InverseGamma(1))
    sigam2 = po.sample('sigma2', dist.InverseGamma(1))


lam_tilde = pm.Normal('lambda_tilde', mu=0, sigma=1)
  lam = pm.Deterministic('lambda', 10 * lam_tilde)
  tau2 = pm.InverseGamma('tau2', alpha= 1.0 , beta= 1.0)
  # prior
  #mu_raw = pm.Normal('mu_raw', mu=0,sigma=1)
  #mu = pm.Deterministic('mu', lam + np.sqrt(tau2)*mu_raw)
  sigma2 = pm.InverseGamma('sigma2', alpha=1.0, beta=1.0)
  # Likelihood
  y_sd = pm.Deterministic('y_sd', np.sqrt(sigma2 + tau2))
  L = pm.Normal('Likelihood', lam, sigma= y_sd, observed=y)
  