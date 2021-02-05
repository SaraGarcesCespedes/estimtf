
<!-- README.md is generated from README.Rmd. Please edit that file -->

# estimtf

<!-- badges: start -->

<!-- badges: end -->

The goal of `estimtf` is to provide functions to find the Maximum
Likelihood estimates of parameters from parametric distributions and
linear regression models using TensorFlow optimizers.

## Installation

You can install the released version of `estimtf` from GitHub with:

``` r
devtools::install_github('SaraGarcesCespedes/estimtf', force=TRUE) 
```

## Example

This is a basic example which shows you how to estimate the mean and standard deviation parameters from the normal distribution using `estimtf`:

``` r
library(estimtf)
#> Loading required package: ggplot2

## Estimation of both normal distrubution parameters
x <- rnorm(n = 1000, mean = 10, sd = 3)

estimation_1 <- dist_estimtf(x, xdist = "Normal", optimizer = "AdamOptimizer",
                             hyperparameters = list(learning_rate = 0.1))
summary(estimation_1)
```
