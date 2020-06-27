// HierNorm.stan
// 
//  yi |mu_i, sigma2 ~ind N(mu_i, sigma^2)
//  mu_i ~iid N(lambda, tau^2)
//  tau^2 ~ Inv-Gamma(1,1)
//  sigma^2 ~ Inv-Gamma(1,1)
//  lambda ~ flat prior
// 
// Note: Non-centered parameterization for mu_i is 
// completed through introducing mu_raw.

data {
  int<lower=1> N; // number of clusters in the population
  real y[N]; // observed cluster means
}
parameters {
  vector[N] mu_raw; //variables for non-centered parameterization
  real lambda;
  // variance are restricted to be positve:
  real<lower=0.00001> tau_sq; // between-cluster variance
  real<lower=0.00001> sigma_sq; // inter-cluster varaince
}
transformed parameters {
  real<lower=0.00001> tau = sqrt(tau_sq);
  real<lower=0.00001> sigma = sqrt(sigma_sq);
  // implies mu ~ normal(lambda, tau)
  vector[N] mu = lambda + tau * mu_raw;
}
model {
  // hyperparameters:
  //  lambda:
  //    non-informative prior in Stan: uniform 
  tau_sq ~ inv_gamma(1, 1);
  // priors:
  sigma_sq ~ inv_gamma(1, 1);
  mu_raw ~ std_normal();
  // likelihood:
  y ~ normal(mu, sigma);
}

