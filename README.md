
<!-- README.md is generated from README.Rmd. Please edit that file -->

<!-- badges: start -->

<!-- [![R-CMD-check](https://github.com/SaraGarcesCespedes/estimtf/workflows/R-CMD-check/badge.svg)](https://github.com/SaraGarcesCespedes/estimtf/actions) -->

[![Licence](https://img.shields.io/badge/licence-GPL--3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
![GitHub R package
version](https://img.shields.io/github/r-package/v/SaraGarcesCespedes/estimtf)
[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/estimtf)](https://cran.r-project.org/package=estimtf)
<!-- [![Travis build status](https://travis-ci.com/SaraGarcesCespedes/estimtf.svg?branch=master)](https://travis-ci.com/SaraGarcesCespedes/estimtf) -->
<!-- badges: end -->

# estimtf <img src="man/figures/logo_sinfondo.png" align="right" height="180" align="right"/>

The `estimtf` package provides functions to find the Maximum Likelihood
Estimates of parameters from probability distributions and linear
regression models using the TensorFlow optimizers.

## Installation

You can install `estimtf` from GitHub. It is recommended to follow these
steps to avoid problems when using the package:

``` r
# Step 1: Install the reticulate package
install.packages("reticulate")
library(reticulate)

# Step 2: Install the tensorflow package
install.packages("tensorflow")
library(tensorflow)

# Step 3: Use the install_tensorflow() funcion to install the TensorFlow module
install_tensorflow()

# Step 4: Confirm that the TensorFlow installation succeded
library(tensorflow)
tf$constant("Hello Tensorflow")

# Step 5: Install the devtools package
install.packages("devtools")

# Step 6: Install and load the estimtf package
devtools::install_github("SaraGarcesCespedes/estimtf", force=TRUE)
library(estimtf)
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
#> Negative log-likelihood: 668.5151 
#> Loss function convergence, 2071 iterations needed. 
#> ---------------------------------------------------
#>      Estimate  Std. Error Z value Pr(>|z|)    
#> mean  10.03652    0.09315   107.7   <2e-16 ***
#> sd     2.94563    0.06517    45.2   <2e-16 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

You can visit the [package
website](https://saragarcescespedes.github.io/estimtf/) to explore the
function reference. Also, to learn more about how to use the functions
of the `estimtf` package, visit this [Colab
notebook](https://colab.research.google.com/drive/1FtMjcwYEF_KqajDwjOqtC2Qh1qzj_zPn?usp=sharing)
that includes examples with multiple distributions and linear regression
models.
