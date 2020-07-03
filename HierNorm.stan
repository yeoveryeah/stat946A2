// HierNorm.stan:
// 
data {
  int<lower=1> N; // number of clusters in the population
  real y[N]; // observed cluster means
}
parameters {
  real lambda_tilda;
  // variance are restricted to be positve:
  real<lower=0.00001> tau_sq; // population variance
  real<lower=0.00001> sigma_sq; // residual varaince
}
transformed parameters {
  // implies lambda ~ normal(0, sd = 10)
  real lambda = 10*lambda_tilda;
  // implies y ~ norma(lambda, sd = y_sd)
  real<lower=0.00001> y_sd = sqrt(tau_sq + sigma_sq);
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

