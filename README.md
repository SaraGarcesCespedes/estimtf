
<!-- README.md is generated from README.Rmd. Please edit that file -->

<!-- badges: start -->

[![R-CMD-check](https://github.com/SaraGarcesCespedes/estimtf/workflows/R-CMD-check/badge.svg)](https://github.com/SaraGarcesCespedes/estimtf/actions)
[![Licence](https://img.shields.io/badge/licence-GPL--3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
![GitHub R package
version](https://img.shields.io/github/r-package/v/SaraGarcesCespedes/estimtf)
[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/estimtf)](https://cran.r-project.org/package=estimtf)
<!-- [![Travis build status](https://travis-ci.com/SaraGarcesCespedes/estimtf.svg?branch=master)](https://travis-ci.com/SaraGarcesCespedes/estimtf) -->
<!-- badges: end -->

# estimtf

The `estimtf` package provides functions to find the Maximum Likelihood
Estimates of parameters from probability distributions and linear
regression models using the TensorFlow optimizers.

## Installation

You can install `estimtf` from GitHub with the following command:

``` r
install.packages('devtools')
devtools::install_github('SaraGarcesCespedes/estimtf', force=TRUE) 
```

## Example

This is a basic example that shows how to estimate the mean and standard
deviation parameters from the normal distribution using the `mle_tf`
function:

``` r
# Load the estimtf package
library(estimtf)

# Estimation of parameters mean and sd from the normal distribution

# Generate a sample from the normal distribution
x <- rnorm(n = 1000, mean = 10, sd = 3)

# Find the MLE of the parameters using the mle_tf function
estimation <- mle_tf(x, 
                     xdist = "Normal", 
                     optimizer = "AdamOptimizer",
                     initparam = list(mean = 0.5, sd = 0.5),
                     hyperparameters = list(learning_rate = 0.1))

# Get the summary of the estimates
summary(estimation)
#> Distribution: Normal 
#> Number of observations: 1000 
#> TensorFlow optimizer: AdamOptimizer 
#> ---------------------------------------------------
#>      Estimate  Std. Error Z value Pr(>|z|)    
#> mean   9.82428    0.09631  102.01   <2e-16 ***
#> sd     3.04552    0.06671   45.65   <2e-16 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```
