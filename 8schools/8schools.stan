// Model is:
// 
//  yi |mu_i, sigma2 ~ind N(mu_i, sigma^2)
//  mu_i ~iid N(lambda, tau^2)
//  tau^2 ~ Inv-Gamma(1,1)
//  sigma^2 ~ Inv-Gamma(1,1)
//  lambda ~ flat prior
// 
//  Model parameters: mu_i, sigma^2 and theta = (lambda, tau)
//      mu are the so-called "random effects".
//      theta are called "hyperparameters".
// variance and s.d are restricted to be positve:

data {
  int<lower=0> N; // sample size
  real y[N]; // estimated treatment of 8 schools
  real<lower=0.00001> sigma[N]; // standard error of 8 schools
}
parameters {
  real mu[N]; 
  real lambda; 
  real<lower=0.00001> tau_sq;
}
transformed parameters {
  real<lower=0.00001> tau = sqrt(tau_sq);
}
model {
  // hyperparameters:
  //  lambda: uniform
  tau_sq ~ inv_gamma(1, 1);
  // priors:
  mu ~ normal(lambda, tau);
  // likelihood:
  y ~ normal(mu, sigma);
}
