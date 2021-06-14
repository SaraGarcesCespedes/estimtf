
<!-- README.md is generated from README.Rmd. Please edit that file -->

<!-- badges: start -->

[![R-CMD-check](https://github.com/SaraGarcesCespedes/estimtf/workflows/R-CMD-check/badge.svg)](https://github.com/SaraGarcesCespedes/estimtf/actions)
[![Licence](https://img.shields.io/badge/licence-GPL--3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
![GitHub R package
version](https://img.shields.io/github/r-package/v/SaraGarcesCespedes/estimtf)

<!-- badges: end -->

# estimtf

The `estimtf` provides functions to find the Maximum Likelihood
estimates of parameters from parametric distributions and linear
regression models using TensorFlow optimizers.

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
library(estimtf)

## Estimation of both normal distrubution parameters
x <- rnorm(n = 1000, mean = 10, sd = 3)

estimation <- mle_tf(x, 
                     xdist = "Normal", 
                     optimizer = "AdamOptimizer",
                     initpara = list(mean = 0.5, sd = 0.5),
                     hyperparameters = list(learning_rate = 0.1))
summary(estimation)
#> Distribution: Normal 
#> Number of observations: 1000 
#> TensorFlow optimizer: AdamOptimizer 
#> ---------------------------------------------------
#>      Estimate  Std. Error Z value Pr(>|z|)    
#> mean    9.9816     0.0932  107.10   <2e-16 ***
#> sd      2.9473     0.0646   45.63   <2e-16 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```
