
#' @title dist_estimtf function
#'
#' @description Function to estimate distributional parameters using TensorFlow
#'
#' @param x a vector with data
#' @param xdist a character indicating the name of the distribution of interest. The default value is \code{'Normal'}
#' @param fixparam a list of the fixed parameters of the distribution of interest. The list must contain the parameters values and names
#' @param initparam a list with initial values of the parameters to be estimated. The list must contain the parameters values and names
#' @param optimizer a character indicating the name of the TensorFlow optimizer to be used in the estimation process The default value is \code{'Adam'}
#' @param hyperparameters a list with the hyperparameters values of the TensorFlow optimizer
#' @param maxiter a positive integer indicating the maximum number of iterations for the optimization algorithm
#' @param tolerance
#' @param eager logical. If TRUE
#' @param comparison logical. If TRUE the
#' @param lower a numeric vector with lower bounds, with the same lenght of argument `initparam`
#' @param upper a numeric vector with upper bounds, with the same lenght of argument `initparam`
#' @param method a character with the name of the optimization routine. \code{nlminb}, \code{optim}, \code{DEoptim} are available
#'
#' @return The output from
#' @export
#'
#' @examples
dist_estimtf <- function(x, xdist = "Normal", fixparam = NULL, initparam = NULL, optimizer, hyperparameters = NULL,
                   maxiter = 1000, tolerance = NULL, eager = TRUE, comparison = FALSE, lower = NULL,
                   upper = NULL, method = "nlminb") {

        library(EstimationTools)
        library(RelDists)
        library(tensorflow)
        library(reticulate)
        library(dplyr)
        library(stringr)
        library(ggplot2)
        # Errors in arguments

        # Error in vector of data x
        if (is.null(x) | length(x) == 0) {
                stop(paste0("For parameter estimation with Maximum Likelihood method, \n",
                            " a vector of data is needed. \n \n"))
        }

        # Error in character xdist
        if (is.null(xdist)) {
                stop("Distribution of x must be specified \n \n")
        }
        if (!is.character(xdist)) {
                stop("'xdist' must be a character \n \n")
        }

        # Defining loss function depending on xdist
        if (xdist != "Poisson" & xdist != "FWE" & xdist != "Instantaneous Failures") {
                dist <- eval(parse(text = paste("tf$compat$v1$distributions$", xdist, sep = "")))
        } else {
                dist <- xdist
        }

        # List of arguments of TensorFlow functions
        if (dist == "Instantaneous Failures" | dist == "Poisson") {
                argumdist <- list(lambda = NULL)
        } else if (dist == "FWE") {
                argumdist <- list(mu = NULL, sigma = NULL)
        } else {
                # Arguments names of tf function
                inspect <- import("inspect")
                argumdist <- inspect$signature(dist)
                argumdist <- argumdist$parameters$copy()
        }


        # Errors in list fixparam
        # Update argumdist. Leaves all the arguments of the TF distribution except the ones that are fixed
        if (!is.null(fixparam)) {
                if (length(match(names(fixparam), names(argumdist))) == 0) {
                        stop(paste0("Names of fixed parameters do not match with the arguments of \n",
                                    dist, " function."))
                } else if (length(match(names(fixparam), names(argumdist))) > 0) {
                        fixed <- match(names(fixparam), names(argumdist))
                        argumdist <- argumdist[-fixed]
                }
        }


        # Calculate number of parameters to be estimated. Remove from argumdist the arguments that are not related with parameters
        if (dist == "Instantaneous Failures" | dist == "Poisson"){
                np <- 1 # number of parameters to be estimated
        } else if (dist == "FWE") {
                np <- 2
        } else {
                arg <- sapply(1:length(argumdist),
                              FUN = function(x) names(argumdist)[x] != "validate_args" & names(argumdist)[x] != "allow_nan_stats" & names(argumdist)[x] != "name" & names(argumdist)[x] != "dtype")
                np <- sum(arg)
        }


        # Errors in list initparam
        if (!is.null(initparam)) {
                if (length(match(names(initparam), names(argumdist))) == 0) {
                        stop(paste0("Names of parameters included in the 'initparam' list do not match with the arguments of ",
                                    dist, " function."))
                } else if (length(match(names(initparam), names(argumdist))) > np) {
                        stop(paste0("Only include in 'initparam' the names of parameters that are not fixed"))
                }
        }

        # List of optimizers
        optimizers <- c("AdadeltaOptimizer", "AdagradDAOptimizer", "AdagradOptimizer", "AdamOptimizer", "GradientDescentOptimizer",
                        "MomentumOptimizer", "RMSPropOptimizer")

        # Error in character for optimizer
        if (!(optimizer %in% optimizers)) {
                stop(paste0("Unidentified optimizer. Select one of the optimizers included in the \n",
                            " following list: ", paste0(optimizers, collapse = ", ")))
        }


        # If the user do not provide initial values for the parameters to be estimated, by default the values will be 0 or 2
        if (is.null(initparam)) {
                initparam <- vector(mode = "list", length = np)
                if (dist == "Instantaneous Failures" | dist == "Poisson" | dist == "FWE"){
                        param <- names(argumdist)
                } else {
                        param <- names(argumdist)[which(names(argumdist)[x] != "validate_args" & names(argumdist)[x] != "allow_nan_stats" & names(argumdist)[x] != "name" & names(argumdist)[x] != "dtype")]
                }
                names(initparam) <- c(param)
                for (i in 1:length(np)) initparam[[i]] <- ifelse(dist == "Instantaneous Failures" | dist == "Poisson", 2.0, 0.0) #SEGURAMENTE SE PUEDE HACER MAS EFICIENTE
        }

        # If the user do not provide tolerance values, by default the values will be .Machine$double.eps
        if (is.null(tolerance)) {
                tolerance <- list(parameters = .Machine$double.eps, loss = .Machine$double.eps, gradients = .Machine$double.eps)
        }

        # Define the TF optimizer depending on the user selection
        opt <- eval(parse(text=paste("tf$compat$v1$train$", optimizer, sep="")))

        # List of arguments of TensorFlow optimizer
        inspect <- import("inspect")
        argumopt <- inspect$signature(opt)
        argumopt <- argumopt$parameters$copy()
        argumopt <- within(argumopt, rm(name)) #remove name argument

        # If the user do not provide values for the hyperparameters, they will take the default values of tensorflow
        if (!is.null(hyperparameters)) {
                if (length(match(names(hyperparameters), names(argumopt))) == 0) {
                        stop(paste0("Names hyperparameters do not match with the hyperparameters of ","TensorFlow ", optimizer, "."))
                }
        } else if (is.null(hyperparameters)) {
                hyperparameters <- vector(mode = "list", length = length(argumopt))
                names(hyperparameters) <- names(argumopt)
                splitarg <- sapply(1:length(argumopt), FUN = function(x) argumopt[[x]] %>% str_split("\\="))
                for (i in 1:length(hyperparameters)) hyperparameters[[i]] <- ifelse(splitarg[[i]][2] == "True" | splitarg[[i]][2] == "False", splitarg[[i]][2], as.numeric(splitarg[[i]][2])) #SE PUEDE HACER MAS EFICIENTE?
        }

        # Estimation process starts

        # With eager execution or disable eager execution
        if (eager == TRUE) {
                res <- eager_estimtf(x, dist, fixparam, linkfun, initparam, opt, hyperparameters, maxiter, tolerance, np)
        } else {
                res <- disableager_estimtf(x, dist, fixparam, linkfun, initparam, opt, hyperparameters, maxiter, tolerance, np)
        }

        # Estimations with other R optimizers using Estimation Tools function

        # List of optimizers available in EstimationTools
        methodsET <- c("nlminb", "optim", "DEoptim")

        # NULL lower and upper
        if (is.null(lower)) lower <- rep(x = -Inf, times = np)
        if (is.null(upper)) upper <- rep(x = Inf, times = np)

        # Error in character for metho
        if (!(method %in% methodsET)) {
                stop(paste0("Unidentified EstimationTools package optimizer. Select one of the optimizers included in the \n",
                            " following list: ", paste0(methodsET, collapse = ", ")))
        }

        if (comparison == TRUE){
                resET <- comparison_estimtf(x, xdist, fixparam, initparam, lower, upper, method)
        }

        return(list(tf = res$final, stderrtf = res$standarderror, esttools = summary(resET)))

}

