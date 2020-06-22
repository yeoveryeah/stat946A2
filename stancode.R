set.seed(42)
N <- 20
sim_lam <- 3
sim_tau2 <- 10
sim_sig2 <- 0.01
x <- rnorm(N, sd = sqrt(sim_tau2))
y <- 3 + 5*x + rnorm(N, sd=sqrt(sim_sig2))
require(rstan)
# Compile the model
hnm_mod <- stan_model(file = "HierNorm.stan") 
# Construct log-posterior: 
hnm_data <- list(N = N, y = y)
# ------------ lp check -------
hnm_fit <- sampling(hnm_mod, data = hnm_data, iter = 1, 
                    verbose = TRUE, chains = 1,
                    init = list(list(lambda = 1, mu = rep(1.5,N), 
                                     tau_sq = 5, sigma_sq = 1)))
# log-post in R
require(invgamma)
logpost <- function(lambda, mu, tau_sq, sigma_sq, y) {
  # Hyper-prior:
  #  no need to include the log of flat prior
  lp <- dinvgamma(tau_sq, shape = 1, scale = 1, log = TRUE)
  # Prior:
  lp <- lp + dinvgamma(sigma_sq, shape = 1, scale = 1, log = TRUE)
  lp <- lp + sum(dnorm(mu, mean = lambda, sd= sqrt(tau_sq), log = TRUE))
  # likelihood:
  lp <- lp + sum(dnorm(y, mean = mu, sd = sqrt(sigma_sq), log = TRUE)) 
  # output: 
  lp
}

# Simulate parameter values
# fixed lambda and tau_sq to get constant difference between
# R and stan
nsim <- 18
Pars <- replicate(n = nsim, expr = {
  list(lambda = rexp(1), mu = rnorm(N), 
       tau_sq = rexp(1)*5, sigma_sq = rexp(1)*2)
}, simplify = FALSE)

# log-posterior calculation in R
lpR <- sapply(1:nsim, function(ii) {
  lambda <- Pars[[ii]]$lambda 
  mu <- Pars[[ii]]$mu 
  tau_sq <- Pars[[ii]]$tau_sq
  sigma_sq <- Pars[[ii]]$sigma_sq
  logpost(lambda = lambda, mu = mu, tau_sq = tau_sq, 
          sigma_sq = sigma_sq, y = hnm_data$y)
})

# log-posterior in Stan
lpStan <- sapply(1:nsim, function(ii) {
  upars <- unconstrain_pars(object = hnm_fit, Pars[[ii]])
  log_prob(object = hnm_fit, upars = upars, adjust_transform = FALSE)
})

lpR-lpStan # return exactly same constant

# ---------- model fit:
nsamples <- 1e6
theta <- c("lambda","tau_sq","sigma_sq", "lp__")
stan_time <- system.time(
  hnm_fit <- sampling(hnm_mod, data = hnm_data, pars= theta,
                      iter = nsamples, control = list(adapt_delta = 0.99)))
