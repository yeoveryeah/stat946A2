import pymc3 as pm 
import numpy as np
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
import seaborn as sb

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

with pm.Model() as hnm_mod:  

  # hyperparameter:
  lam_tilde = pm.Normal('lambda_tilde', mu=0, sigma=1)
  lam = pm.Deterministic('lambda', 10 * lam_tilde)
  tau2 = pm.InverseGamma('tau2', alpha= 1.0 , beta= 1.0)
  # prior
  sigma2 = pm.InverseGamma('sigma2', alpha=1.0, beta=1.0)
  # Likelihood
  y_sd = pm.Deterministic('y_sd', np.sqrt(sigma2 + tau2))
  L = pm.Normal('Likelihood', lam, sigma= y_sd, observed=y)
  
# Model fitting: inference
# PyMC3 automatically initializes NUTS to reasonable values based on the
#   variance of the samples obtained during a tuning phase.
with hnm_mod:
    #db = pm.backends.Text('hnm_pymc3')
    trace_hnm = pm.sample(draws= nsamples/2, tune=nsamples/2, #trace=db,  
                          target_accept=0.99, chains=4, cores=1)
    pm.traceplot(trace_hnm, var_names = ["lambda", "tau2", "sigma2"], compact=True, legend=True)
    plt.show()


#trarr = traceplot(trace_hnm)
#fig = plt.gcf() # to get the current figure...
#fig.savefig("disaster.png") # and save it directly

# Traceplot
#pm.traceplot(trace_hnm,  vars=['lambda', 'tau2', 'sigma2'])
# Posterior
#lam_post = trace_hnm.get_values['lambda', burn=nsamples/2, combine = True]

#tau2_post = trace_hnm.get_values['tau2', burn=nsamples/2, combine = False]
#sigma2_post =trace_hnm.get_values['sigma2', burn=nsamples/2, combine = False]