#' Log-Posterior Function
#'
#' @description  `logpost` returns the log-posterior of the hierarchical model given the parameter values and observations
#'  
#' @param lambda A real number, the hyperparameter
#' @param mu_raw A vector of reals at the same length of y
#' @param tau_sq A positive number, the hyperparameter
#' @param sigma_sq A positive number
#' @param y A vector of reals 
#' @return The log-posterior values
#' @details The hyper-prior are given by tau_sq ~ InverseGamma(1,1) and lambda ~ Uniform, where the constant uniform probability are eliminated in calculating the log-posterior.
#' The prior are given by mu_raw ~ N(0,1 ) and sigma_sq InverseGamma(1,1), while the non-centered parameterization are implemented here. 
#' The likelihood are given by the Normal(mu, sigma_sq)
#' @import invgamma 
#' @export
logpost <- function(lambda, mu_raw, tau_sq, sigma_sq, y) {
  require(invgamma)
  # Hyper-prior:
  lp <- dinvgamma(tau_sq, shape = 1, scale = 1, log = TRUE)
  # Prior:
  lp <- lp + dinvgamma(sigma_sq, shape = 1, scale = 1, log = TRUE)
  lp <- lp + sum(dnorm(mu_raw, log = TRUE))
  # likelihood:
  mu <- lambda + sqrt(tau_sq) * mu_raw 
  lp <- lp + sum(dnorm(y, mean = mu, sd = sqrt(sigma_sq), log = TRUE)) 
  # output: 
  lp
}