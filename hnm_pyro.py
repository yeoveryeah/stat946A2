# hnm_pyro.py: 
import numpyro as po 
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, util
from jax import random
import jax.numpy as np
import timeit

import matplotlib.pyplot as plt
import arviz as az
import os

# 1. Data
#  The true parameters are given as: 
N = 20
true_lam = 3
true_tau2 = 1.5 
true_sig2 = 1.2
# Loda the data
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(THIS_FOLDER, 'data.txt')
d = open(data_file)
y = np.array([float(line.strip('\n')) for line in d.readlines()])

# 2. Build the model
def hnm_mod(N, y = None):
    """ Estimate the posterior estiamtes of the hierarchical model given below:
        y_i | mu_i, sigma_sq ~ N(mu_i, var = sigma_sq)
        mu_i ~ N(lambda, var = tau_sq)
        lambda ~ N(0,var = 100)
        sigma_sq, tau_sq ~ Inverse-Gamma(1,1)

    By taking reparameterization, it's equivalent to 
        lam_tilde ~ N(0,1), lambda = 10*lam_tilde
        tau_sq, sigma_sq ~ Inverse-Gamma(1,1)
        y_sd = sqrt(tau_sq + sigma_sq)
        y ~ N(lambda, y_sd)

    Args:
        N (int): the length of y
        y (array): the observed y

    Returns:
        Float: Prediction of y by the model 
    """
    # hyper-prior
    lam_tilde = po.sample('lam_tilde', dist.Normal(0., 1.))
    lam = po.deterministic('lambda', 10*lam_tilde)
    tau_sq = po.sample('tau_sq', dist.InverseGamma(concentration=1,rate=1))
    # prior
    sigma_sq = po.sample('sigma_sq', dist.InverseGamma(concentration=1, rate=1))
    # Likelihood
    with po.plate("N", N):
        y_sd = po.deterministic('y_sd', np.sqrt(sigma_sq + tau_sq))
        po.sample('Likelihood', dist.Normal(loc=lam, scale=y_sd), obs=y)


# 3. Check the correctness of the model:
#  the log_density function will return the log posterior values at the given parameter values
#   and a corresponding model trace
log_prob = util.log_density(hnm_mod, model_args=(), model_kwargs={'N':N,'y':y}, 
params={'lam_tilde':1.8,'tau_sq':4.6, 'sigma_sq':4.})
print(log_prob[0])

# 4. NUTS sampling 
nuts_kernel = NUTS(hnm_mod, target_accept_prob=0.99) # Assign sampler to MCMC
mcmc = MCMC(nuts_kernel, num_warmup = 5000, num_samples=5000, num_chains=4)
rng_key = random.PRNGKey(123) #Random number generator key to be used for sampling
mcmc.run(rng_key, N=N, y = y) #run the NUTS sampler and collect the samples
post_sample = mcmc.get_samples(group_by_chain=True) # Extrace the samples and separate by chains

# 5. Traceplot
az.plot_trace(post_sample, var_names=['lambda', 'tau_sq', 'sigma_sq'],combined=False, legend=True)

# 6. Posterior histograms
az.plot_posterior(post_sample, var_names=['lambda', 'tau_sq', 'sigma_sq'], kind='hist', point_estimate='mean', ref_val=[true_lam, true_tau2, true_sig2], hdi_prob='hide', credible_interval=None)