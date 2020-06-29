import numpyro as po 
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from jax import random
import numpy as np
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

def hnm_mod(N, y = None):
    # hyper-prior
    lam_tilde = po.sample('lam_tilde', dist.Normal(0,1))
    tau_sq = po.sample('tau_sq', dist.InverseGamma(1))
    # prior
    sigma_sq = po.sample('sigma_sq', dist.InverseGamma(1))
    # Likelihood
    y_sd = po.deterministic('y_sd', np.sqrt(sigma_sq + tau_sq))
    with po.plate('N', N):
      L = po.sample('Likelihood', dist.Normal(lam, sigma = y_sd), obs=y)

nuts_kernel = NUTS(hnm_mod, target_accept_prob=0.99, cha)
mcmc = MCMC(nuts_kernel, num_warmup = nsamples/2, num_samples = nsamples/2, num_chains= 4)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, N, y = y)
posterior_samples = mcmc.get_samples(group_by_chain=TRue)


  