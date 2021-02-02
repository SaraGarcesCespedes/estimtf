#' @title dist_estimtf function
#'
#' @description Function to compute the Maximum Likelihood Estimators of distributional parameters using TensorFlow.
#'
#' @author Sara Garces Cespedes
#'
#' @param x a vector containing the data to be fitted.
#' @param xdist a character indicating the name of the distribution of interest. The default value is \code{'Normal'}.
#' @param fixparam a list of the fixed parameters of the distribution of interest. The list must contain the parameters values and names.
#' @param initparam a list with initial values of the parameters to be estimated. The list must contain the parameters values and names.
#' @param optimizer a character indicating the name of the TensorFlow optimizer to be used in the estimation process The default value is \code{'AdamOptimizer'}. The available optimizers are:
#' \code{"AdadeltaOptimizer"}, \code{"AdagradDAOptimizer"}, \code{"AdagradOptimizer"}, \code{"AdamOptimizer"}, \code{"GradientDescentOptimizer"},
#' \code{"MomentumOptimizer"} and \code{"RMSPropOptimizer"}.
#' @param hyperparameters a list with the hyperparameters values of the TensorFlow optimizer. FALTA DETALLES
#' @param maxiter a positive integer indicating the maximum number of iterations for the optimization algorithm.
#'
#' @return This function returns the estimates and standard errors of parameters from statistical distributions as well as
#' some information of the optimization process like the number of iterations needed for convergence.
#'
#' @details \code{dist_estimtf} computes the log-likelihood function of the distribution specified in
#' \code{xdist} and finds the distributional parameters that maximizes it using TensorFlow.
#'
#' @importFrom stringr str_split
#' @importFrom dplyr %>%
#' @importFrom dplyr select
#' @importFrom dplyr all_of
#' @importFrom stats na.omit
#' @importFrom stats terms
#' @importFrom stats model.matrix
#' @importFrom stats delete.response
#' @importFrom stats printCoefmat
#' @importFrom stats pnorm
#' @import tensorflow
#' @import tfprobability
#'
#'
#' @examples
#' #-------------------------------------------------------------
#' # Estimation of both normal distrubution parameters
#' x <- rnorm(n = 1000, mean = 10, sd = 3)
#'
#' estimation_1 <- dist_estimtf(x, xdist = "Normal", optimizer = "AdamOptimizer",
#'                            hyperparameters = list(learning_rate = 0.1))
#'
#' summary(estimation_1)
#'
#' #-------------------------------------------------------------
#' # Estimation with one fixed parameter
#' x <- rnorm(n = 1000, mean = 10, sd = 3)
#'
#' estimation_2 <- dist_estimtf(x, xdist = "Normal", optimizer = "AdamOptimizer",
#'                            hyperparameters = list(learning_rate = 0.1),
#'                            fixparam = list(loc = 10))
#'
#' summary(estimation_2)
#'
#'
#' @export
dist_estimtf <- function(x, xdist = "Normal", fixparam = NULL, initparam = NULL, optimizer = "AdamOptimizer", hyperparameters = NULL,
                         maxiter = 10000) {

        #suppressMessages(library(EstimationTools)) ; suppressMessages(library(RelDists)) ;
        #library(tensorflow)
        #library(reticulate)
        #library(dplyr)
        #library(stringr)
        #library(ggplot2)
        #library(tfprobability)
        #library(purrr)

        call <- match.call()

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
        distdisponibles <- c("Normal", "Poisson", "Uniform", "Gamma", "LogNormal", "Weibull", "Exponential",
                             "Beta", "Cauchy", "StudentT", "Chi2", "Logistic","FWE", "InstantaneousFailures",
                             "DoubleExponential")
        distnotf <- c("FWE", "InstantaneousFailures", "DoubleExponential")

        if (!(xdist %in% distdisponibles)) {
                stop(paste0("The distribution is not available. The following are the \n",
                            " available distributions: ", paste0(distdisponibles, collapse = ", ")))
        }

        if (!(xdist %in% distnotf)) {
                dist <- eval(parse(text = paste("tfprobability::tfp$distributions$", xdist, sep = "")))
        } else {
                dist <- xdist
        }

        # List of arguments of TensorFlow functions
        if (xdist %in% distnotf) {
                argumdist <- arguments(dist)
        } else {
                # Arguments names of tf function
                inspect <- reticulate::import("inspect")
                argumdist <- inspect$signature(dist)
                argumdist <- argumdist$parameters$copy()
        }

        # Errors in list fixparam
        # Update argumdist. Leaves all the arguments of the TF distribution except the ones that are fixed
        if (!is.null(fixparam)) {
                if (length(na.omit(match(names(fixparam), names(argumdist)))) == 0) {
                        stop(paste0("Names of parameters included in the 'fixparam' list do not match with the parameters of the \n",
                                    dist, " distribution"))
                } else if (length(na.omit(match(names(fixparam), names(argumdist)))) > 0) {
                        fixed <- match(names(fixparam), names(argumdist))
                        argumdist <- argumdist[-fixed]
                }
        }


        # Calculate number of parameters to be estimated. Remove from argumdist the arguments that are not related with parameters
        if (xdist %in% distnotf){
                np <- length(argumdist) # number of parameters to be estimated
        } else {
                arg <- sapply(1:length(argumdist),
                              FUN = function(x) names(argumdist)[x] != "validate_args" & names(argumdist)[x] != "allow_nan_stats" &
                                      names(argumdist)[x] != "name" & names(argumdist)[x] != "dtype" &
                                      names(argumdist)[x] != "interpolate_nondiscrete" & names(argumdist)[x] != "log_rate")
                np <- sum(arg)
                argumdist <- argumdist[arg]
        }


        # Errors in list initparam
        if (!is.null(initparam)) {
                if (all(names(initparam) %in% names(argumdist)) == FALSE) {
                        stop(paste0("Some or all of the parameters included in the 'initparam' list do not match with the arguments of ",
                                    all.vars(ydist)[2], " distribution."))
                } else if (length(na.omit(match(names(initparam), names(argumdist)))) > np) {
                        stop(paste0("Only include in 'initparam' the names of parameters that are not fixed"))
                } else {
                        providedvalues <- match(names(initparam), names(argumdist))
                        namesprovidedvalues <- names(initparam)
                        missingvalues <- argumdist[-providedvalues]
                        initparam <- append(initparam, rep(1.0, length(missingvalues))) #valor de 1 a los parametros que no me dieron initparam
                        names(initparam) <- c(namesprovidedvalues, names(missingvalues))
                }
        }


        # If the user do not provide initial values for the parameters to be estimated, by default the values will be 1 or 2
        if (is.null(initparam)) {
                initparam <- vector(mode = "list", length = np)
                initparam <- lapply(1:np,
                                    FUN = function(i) initparam[[i]] <- ifelse(dist == "InstantaneousFailures" | dist == "Poisson", 2.0, 1.0))
                names(initparam) <- names(argumdist)
        }

        # order of initparam and par_names(names argumdist) must be the same
        initparam <- initparam[names(argumdist)]

        # List of optimizers
        optimizers <- c("AdadeltaOptimizer", "AdagradDAOptimizer", "AdagradOptimizer", "AdamOptimizer", "GradientDescentOptimizer",
                        "MomentumOptimizer", "RMSPropOptimizer")
        # Error in character for optimizer
        if (!(optimizer %in% optimizers)) {
                stop(paste0("Unidentified optimizer. Select one of the following optimizers: \n",
                             paste0(optimizers, collapse = ", ")))
        }


        # Value for tolerance
        tolerance <- list(parameters = .Machine$double.eps, loss = .Machine$double.eps, gradients = .Machine$double.eps)


        # Define the TF optimizer depending on the user selection
        opt <- eval(parse(text=paste("tensorflow::tf$compat$v1$train$", optimizer, sep="")))

        # List of arguments of TensorFlow optimizer
        inspect <- reticulate::import("inspect")
        argumopt <- inspect$signature(opt)
        argumopt <- argumopt$parameters$copy()
        argumopt <- within(argumopt, rm(name)) #remove name argument
        argumopt <- within(argumopt, rm(use_locking))

        # If the user do not provide values for the hyperparameters, they will take the default values of tensorflow
        if (!is.null(hyperparameters)) {
                if (length(na.omit(match(names(hyperparameters), names(argumopt)))) == 0) {
                        stop(paste0("The hyperparameters included in the list do not match with the hyperparameters of ", optimizer, ": \n",
                                    paste0(argumopt, collapse = ", ")))
                }
        } else if (is.null(hyperparameters)) {
                hyperparameters <- vector(mode = "list", length = length(argumopt))
                names(hyperparameters) <- names(argumopt)
                splitarg <- sapply(1:length(argumopt), FUN = function(x) argumopt[[x]] %>% str_split("\\="))
                hyperparameters <- lapply(1:length(hyperparameters),
                                          FUN = function(i) hyperparameters[[i]] <- ifelse(splitarg[[i]][2] == "True" | splitarg[[i]][2] == "False",
                                                                                           splitarg[[i]][2], as.numeric(splitarg[[i]][2])))
        }

        # Estimation process starts

        # With disable eager execution
        res <- disableagerdist(x, dist, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np, distnotf, xdist, optimizer)


        result <- list(tf = res$results, vvoc = res$vcov, stderrtf = res$standarderror,
                       outputs = res$outputs, distribution = xdist, optimizer = optimizer, call = call)
        class(result) <- "MLEtf"
        return(result)

}

#------------------------------------------------------------------------
# List of arguments for distributions not included in TF ----------------
#------------------------------------------------------------------------

arguments <- function(dist) {

        listarguments <- list(InstantaneousFailures = list(lambda = NULL),
                              Weibull = list(shape = NULL, scale = NULL),
                              DoubleExponential = list(loc = NULL, scale = NULL))

        return(listarguments[[dist]])

}
