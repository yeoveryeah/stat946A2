#' Log-Posterior Function
#'
#' @description  `logpost` returns the log-posterior of the hierarchical model given the parameter values and observations
#'  
#' @param lambda_tilda  real number, the hyperparameter
#' @param tau_sq  positive number, the hyperparameter
#' @param sigma_sq positive number
#' @param y vector of reals 
#' @return The log-posterior values
#' @details The hyper-prior are given by tau_sq ~ InverseGamma(1,1) and lambda_tilda ~ N(0,1) such that lambda ~ N(0,100).
#' The prior are given by sigma_sq ~ InverseGamma(1,1);
#' The likelihood are given by the y ~ Normal(lambda, sqrt(sigma_sq + tau_sq))
#' @import invgamma 
#' @export
logpost <- function(lambda_tilda, tau_sq, sigma_sq, y) {
  require(invgamma)
  # Hyper-prior:
  lp <- dnorm(lambda_tilda, log = TRUE)
  lp <- lp + dinvgamma(tau_sq, shape = 1, scale = 1, log = TRUE)
  # Prior:
  lp <- lp + dinvgamma(sigma_sq, shape = 1, scale = 1, log = TRUE)
  # likelihood:
  lambda <- 10*lambda_tilda # implies lambda ~ N(0,100)
  y_sd <- sqrt(sigma_sq+tau_sq)
  lp <- lp + sum(dnorm(y, mean = lambda, sd = y_sd, log = TRUE)) 
  # output: 
  lp
}