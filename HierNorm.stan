// HierNorm.stan
// 
//  yi |mu_i, sigma2 ~ind N(mu_i, sigma^2)
//  mu_i ~iid N(lambda, tau^2)
//  tau^2 ~ Inv-Gamma(1,1)
//  sigma^2 ~ Inv-Gamma(1,1)
//  lambda ~ N(0,10^2)
// 
// Note: Non-centered parameterization is employed.

data {
  int<lower=1> N; // number of clusters in the population
  real y[N]; // observed cluster means
}
parameters {
  real lambda_tilda;
  // variance are restricted to be positve:
  real<lower=0.00001> tau_sq; // cross-cluster variance
  real<lower=0.00001> sigma_sq; // inter-cluster varaince
}
transformed parameters {
  real<lower=0.00001> y_sd = sqrt(tau_sq + sigma_sq);
  real<lower=0.00001> tau = sqrt(tau_sq);
  // implies lambda ~ normal(0, sd = 10)
  real lambda = 10*lambda_tilda;
}
model {
  // hyper-priors:
  tau_sq ~ inv_gamma(1, 1);
  lambda_tilda ~ std_normal();
  // priors:
  sigma_sq ~ inv_gamma(1, 1);
  // likelihood:
  y ~ normal(lambda, y_sd);
}

