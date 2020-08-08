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
#' @param optimizer a character indicating the name of the TensorFlow optimizer to be used in the estimation process The default value is \code{'AdamOptimizer'}.
#' @param hyperparameters a list with the hyperparameters values of the TensorFlow optimizer. FALTA DETALLES
#' @param maxiter a positive integer indicating the maximum number of iterations for the optimization algorithm.
#' @param eager logical. If \code{TRUE}, the estimation process is performed in the eager execution environment. The default value is \code{TRUE}.
#'
#' @return This function returns the estimates and standard errors of distributional parameters as well as
#' some information of the optimization process like the number of iterations needed for convergence.
#'
#' @details \code{dist_estimtf} computes the log-likelihood function of the distribution specified in
#' \code{xdist} and finds the distributional parameters that maximizes it using TensorFlow.
#'
#' @importFrom stringr str_split
#'
#'
#' @examples
#' #-------------------------------------------------------------
#' # Estimation of both normal distrubution parameters
#' x <- rnorm(n = 1000, mean = 10, sigma = 3)
#'
#' estimation_1 <- dist_estimtf(x, xdist = "Normal", optimizer = "AdamOptimizer",
#'                            hyperparameters = list(learning_rate = 0.1))
#'
#' summary(estimation_1)
#'
#' #-------------------------------------------------------------
#' # Estimation with one fixed parameter
#' x <- rnorm(n = 1000, mean = 10, sigma = 3)
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
                   maxiter = 10000, eager = TRUE) {

        suppressMessages(library(EstimationTools)) ; suppressMessages(library(RelDists)) ;
        suppressMessages(library(tensorflow)) ; suppressMessages(library(reticulate))
        suppressMessages(library(dplyr)) ; suppressMessages(library(stringr))

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
        distnotf <- c("Poisson", "FWE", "InstantaneousFailures", "Weibull", "Cauchy",
                      "DoubleExponential", "Geometric", "LogNormal")
        if (!(xdist %in% distnotf)) {
                dist <- eval(parse(text = paste("tf$compat$v1$distributions$", xdist, sep = "")))
        } else {
                dist <- xdist
        }

        # List of arguments of TensorFlow functions
        if (xdist %in% distnotf) {
                argumdist <- arguments(dist)
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
        if (xdist %in% distnotf){
                np <- length(argumdist) # number of parameters to be estimated
        } else {
                arg <- sapply(1:length(argumdist),
                              FUN = function(x) names(argumdist)[x] != "validate_args" & names(argumdist)[x] != "allow_nan_stats" & names(argumdist)[x] != "name" & names(argumdist)[x] != "dtype")
                np <- sum(arg)
                argumdist <- argumdist[arg]
        }


        # Errors in list initparam
        if (!is.null(initparam)) {
                if (length(match(names(initparam), names(argumdist))) == 0) {
                        stop(paste0("Names of parameters included in the 'initparam' list do not match with the parameters of the ",
                                    dist, " distribution"))
                } else if (length(match(names(initparam), names(argumdist))) > np) {
                        stop(paste0("Only include in 'initparam' the names of parameters that are not fixed"))
                } else if (length(match(names(initparam), names(argumdist))) > 0 & length(match(names(initparam), names(argumdist))) < np) {
                        providedvalues <- match(names(initparam), names(argumdist))
                        namesprovidedvalues <- names(initparam)
                        missingvalues <- argumdist[-providedvalues]
                        initparam <- append(initparam, rep(1.0, length(missingvalues)))
                        names(initparam) <- c(namesprovidedvalues, names(missingvalues))
                }
        }

        # If the user do not provide initial values for the parameters to be estimated, by default the values will be 0 or 2
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
                stop(paste0("Unidentified optimizer. Select one of the optimizers included in the \n",
                            " following list: ", paste0(optimizers, collapse = ", ")))
        }


        # Value for tolerance
        tolerance <- list(parameters = .Machine$double.eps, loss = .Machine$double.eps, gradients = .Machine$double.eps)


        # Define the TF optimizer depending on the user selection
        opt <- eval(parse(text=paste("tf$compat$v1$train$", optimizer, sep="")))

        # List of arguments of TensorFlow optimizer
        inspect <- import("inspect")
        argumopt <- inspect$signature(opt)
        argumopt <- argumopt$parameters$copy()
        argumopt <- within(argumopt, rm(name)) #remove name argument
        argumopt <- within(argumopt, rm(use_locking))

        # If the user do not provide values for the hyperparameters, they will take the default values of tensorflow
        if (!is.null(hyperparameters)) {
                if (length(match(names(hyperparameters), names(argumopt))) == 0) {
                        stop(paste0("Names hyperparameters do not match with the hyperparameters of ","TensorFlow ", optimizer, "."))
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

        # With eager execution or disable eager execution
        if (eager == TRUE) {
                res <- eagerdist(x, dist, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np, distnotf, xdist, optimizer)
        } else {
                res <- disableagerdist(x, dist, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np, distnotf, xdist, optimizer)
        }

        result <- list(tf = res$results, vvoc = res$vcov, stderrtf = res$standarderror,
                       outputs = res$outputs, distribution = xdist, optimizer = optimizer, call = call)
        class(result) <- "MLEtf"
        return(result)

}

#------------------------------------------------------------------------
# List of arguments for distributions not included in TF ----------------
#------------------------------------------------------------------------

arguments <- function(dist) {

        listarguments <- list(Poisson = list(lambda = NULL), FWE = list(mu = NULL, sigma = NULL),
                              InstantaneousFailures = list(lambda = NULL),
                              Weibull = list(shape = NULL, scale = NULL),
                              Cauchy = list(loc = NULL, scale = NULL), Geometric = list(prob = NULL),
                              DoubleExponential = list(loc = NULL, scale = NULL),
                              LogNormal = list(meanlog = NULL, sdlog = NULL))

        return(listarguments[[dist]])

}
