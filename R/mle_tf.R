#' @title mle_tf function
#'
#' @description Function to find the Maximum Likelihood Estimates of distributional parameters using TensorFlow.
#'
#' @author Sara Garcés Céspedes \email{sgarcesc@unal.edu.co}
#'
#' @param x a vector containing the data to be fitted.
#' @param xdist a character indicating the name of the distribution of interest. The default value is \code{'Normal'}.
#' The available distributions are: \code{Normal}, \code{Poisson}, \code{Binomial}, \code{Weibull}, \code{Exponential}, \code{LogNormal}, \code{Beta} and \code{Gamma}.
#' If you want to estimate parameters from a distribution different to the ones mentioned above, you must provide the
#' name of an object of class function that contains its probability mass/density function. This \code{R} function must not contain curly brackets
#' other than those that enclose the function.
#' @param fixparam a list containing the fixed parameters of the distribution of interest only if they exist. The parameters values and names must be specified in the list.
#' @param initparam a list with initial values of the parameters to be estimated. The list must contain the parameters values and names.
#' @param optimizer a character indicating the name of the TensorFlow optimizer to be used in the estimation process The default value is \code{'AdamOptimizer'}. The available optimizers are:
#' \code{"AdadeltaOptimizer"}, \code{"AdagradDAOptimizer"}, \code{"AdagradOptimizer"}, \code{"AdamOptimizer"}, \code{"GradientDescentOptimizer"},
#' \code{"MomentumOptimizer"} and \code{"RMSPropOptimizer"}.
#' @param hyperparameters a list with the hyperparameters values of the selected TensorFlow optimizer. If the hyperparameters are not specified, their default values
#' will be used in the oprimization process. For more details of the hyperparameters go to this URL:
#' \href{https://www.tensorflow.org/api_docs/python/tf/compat/v1/train}{https://www.tensorflow.org/api_docs/python/tf/compat/v1/train}
#' @param maxiter a positive integer indicating the maximum number of iterations for the optimization algorithm. The default value is \code{10000}.
#' @param tolerance a small positive number. When the difference between the loss value or the parameters values from one iteration to another is lower
#' than this value, the optimization process stops. The default value is \code{.Machine$double.eps}.
#'
#' @return This function returns the estimates, standard errors, Z-score and p-values of significance tests of the parameters from the distribution of interest as well as
#' some information of the optimization process like the number of iterations needed for convergence.
#'
#' @details \code{mle_tf} computes the log-likelihood function of the distribution specified in
#' \code{xdist} and finds the values of the parameters that maximizes this function using the TensorFlow optimizer
#' specified in \code{optimizer}.
#'
#'  The \code{R} function that contains the probability mass/density function must not contain curly brackets. The only curly brackets that the function can contain are those that enclose the function,
#'  that is, those that define the beginning and end of the \code{R} function.
#'
#' @note The \code{summary, print, plot_loss} functions can be used with a \code{mle_tf} object.
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
#' @import reticulate
#'
#'
#' @examples
#' #-----------------------------------------------------------------------------
#' # Estimation of parameters mean and sd of the normal distribution
#'
#' # Vector with the data to be fitted
#' x <- rnorm(n = 1000, mean = 10, sd = 3)
#'
#' # Use the mle_tf function
#' estimation_1 <- mle_tf(x,
#'                        xdist = "Normal",
#'                        optimizer = "AdamOptimizer",
#'                        initparam = list(mean = 1.0, sd = 1.0),
#'                        hyperparameters = list(learning_rate = 0.1))
#'
#' # Get the summary of the estimates
#' summary(estimation_1)
#'
#' #-----------------------------------------------------------------------------
#' # Estimation of parameter lambda of the Instantaneous Failures distribution
#'
#' # Create an R function that contains the probability density function
#' pdf <- function(X, lambda) { (1 / ((lambda ^ 2) * (lambda - 1))) *
#'                              (lambda^2 + X - 2*lambda) * exp(-X/lambda) }
#'
#' # Vector with the data to be fitted
#' x <-  c(3.4, 0.0, 0.0, 15.8, 232.8, 8.8, 123.2, 47, 154, 103.2, 89.8,  12.2)
#'
#' # Use the mle_tf function
#' estimation_2 <- mle_tf(x = x,
#'                        xdist = pdf,
#'                        initparam = list(lambda = rnorm(1, 5, 1)),
#'                        optimizer = "AdamOptimizer",
#'                        hyperparameters = list(learning_rate = 0.1),
#'                        maxiter = 10000)
#'
#' # Get the summary of the estimates
#' summary(estimation_2)
#'
#' @export
mle_tf <- function(x, xdist = "Normal", fixparam = NULL, initparam, optimizer = "AdamOptimizer", hyperparameters = NULL,
                   maxiter = 10000, tolerance = .Machine$double.eps) {

        call <- match.call()


        # Errors in arguments

        # missing arguments
        if (missing(x)) {
                stop(paste0("Argument 'x' is missing, with no default"))
        } else if (missing(initparam)) {
                stop(paste0("Argument 'initparam' is missing, with no default"))
        }

        # Error in vector of data x
        if (is.null(x) | length(x) == 0) {
                stop(paste0("For parameter estimation with Maximum Likelihood method, \n",
                            " a vector of data is needed. \n \n"))
        }

        # Error in character xdist
        if (is.null(xdist)) {
                stop("Distribution of x must be specified. \n \n")
        }

        if (!is.character(xdist) & !is.function(xdist)) {
                stop("'xdist' must be a character or a function. \n \n")
        }

        if (!inherits(initparam, "list")) {
                stop(paste0("'initparam' must be a nonempty list containing the initial values, \n",
                            "of the parameters to be estimated. \n \n"))
        }

        if (is.character(xdist)) {
                # Import tensorflow_probability module
                #tf_prob <- reticulate::import("tensorflow_probability")

                # Defining loss function depending on xdist
                distdisponibles <- c("Normal", "Poisson", "Gamma", "LogNormal", "Weibull", "Exponential",
                                     "Beta", "Binomial")
                distnotf <- c("FWE", "InstantaneousFailures", "DoubleExponential")

                if (!(xdist %in% distdisponibles)) {
                        stop(paste0("The distribution is not available. The following are the \n",
                                    " available distributions: ", paste0(distdisponibles, collapse = ", ")))
                }

                if (!(xdist %in% distnotf)) {
                        dist <- eval(parse(text = paste("tfprobability::tfp$distributions$", xdist, sep = "")))
                        #dist <- eval(parse(text = paste("tf_prob$distributions$", xdist, sep = "")))
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

                        # change names of parameters to match TF parameters
                        names_param <- names(fixparam)
                        names_new <- vector(mode = "numeric", length = length(names_param))
                        names_new <- sapply(1:length(names_param), FUN = function(i) names_new[i] <- parameter_name_tf(names_param[i], xdist))
                        names(fixparam) <- names_new

                        if (length(na.omit(match(names(fixparam), names(argumdist)))) == 0) {
                                stop(paste0("Names of parameters included in the 'fixparam' list do not match with the parameters of the \n",
                                            xdist, " distribution"))
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
                                              names(argumdist)[x] != "interpolate_nondiscrete" & names(argumdist)[x] != "log_rate" &
                                              names(argumdist)[x] != "force_probs_to_zero_outside_support" & names(argumdist)[x] != "logits")
                        np <- sum(arg)
                        argumdist <- argumdist[arg]
                }

                # Errors in list initparam
                if (!is.null(initparam)) {

                        # change names of parameters to match TF parameters
                        names_param <- names(initparam)
                        names_new <- vector(mode = "numeric", length = length(names_param))
                        names_new <- sapply(1:length(names_param), FUN = function(i) names_new[i] <- parameter_name_tf(names_param[i], xdist))
                        names(initparam) <- names_new

                        if (all(names(initparam) %in% names(argumdist)) == FALSE) {
                                stop(paste0("Some or all of the parameters included in the 'initparam' list do not match with the arguments of ",
                                            " the provided distribution."))
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
                # if (is.null(initparam)) {
                #         initparam <- vector(mode = "list", length = np)
                #         initparam <- lapply(1:np,
                #                             FUN = function(i) initparam[[i]] <- ifelse(dist == "InstantaneousFailures" | dist == "Poisson", 2.0, 1.0))
                #         names(initparam) <- names(argumdist)
                # }

                # order of initparam and par_names(names argumdist) must be the same
                initparam <- initparam[names(argumdist)]

        } else {
                # List of arguments FDP
                fdp <- xdist
                arguments <- formals(fdp)
                arguments <- as.list(arguments)

                # eliminar x de lista vartotal
                #arguments$X <- NULL
                arguments <- as.list(arguments)
                if ("X" %in% names(arguments)) {
                        arguments$X <- NULL
                } else if ("x" %in% names(arguments)) {
                        arguments$x <- NULL

                } else {
                        message('Caught an error!')
                        stop(paste0('Argument "x" is missing in the probability mass/density function.'))
                }

                # remove fixed parameters
                # arguments_fixed <- vector(mode = "list", length = length(arguments))
                # arguments_fixed <- lapply(1:length(arguments), FUN = function(i) arguments_fixed[[i]] <- ifelse(is.numeric(arguments[[i]]), arguments[[i]], NA))
                # names(arguments_fixed) <- names(arguments)
                # fixparam <- arguments_fixed %>% purrr::discard(is.na)
                #
                # # Calculate number of parameters to be estimated
                # argumdist <- arguments_fixed %>% purrr::discard(is.numeric)
                # np <- length(argumdist)

                # remove fixed parameters
                if (!is.null(fixparam)) {

                        if (length(na.omit(match(names(arguments), names(fixparam)))) == 0) {
                                stop(paste0("Names of fixed parameters do not match with the arguments of \n",
                                            all.vars(ydist)[2], " distribution"))
                        } else if (length(na.omit(match(names(fixparam), names(arguments)))) > 0) {
                                fixed <- match(names(fixparam), names(arguments))
                                argumdist <- arguments[-fixed]
                        }
                } else {
                        argumdist <- arguments
                }

                np <- length(argumdist)

                # Errors in list initparam
                if (!is.null(initparam)) {
                        if (all(names(initparam) %in% names(argumdist)) == FALSE) {
                                stop(paste0("Some or all of the parameters included in the 'initparam' list do not match with the arguments of ",
                                            "the fdp."))
                        } else if (length(na.omit(match(names(initparam), names(argumdist)))) > np) {
                                stop(paste0("Only include in 'initparam' the names of parameters that are not fixed"))
                        } else {
                                providedvalues <- match(names(initparam), names(argumdist))
                                namesprovidedvalues <- names(initparam)
                                missingvalues <- argumdist[-providedvalues]
                                initparam <- append(initparam, rep(0.0, length(missingvalues))) #valor de 1 a los parametros que no me dieron initparam
                                names(initparam) <- c(namesprovidedvalues, names(missingvalues))
                        }
                }


                # If the user do not provide initial values for the parameters to be estimated, by default the values will be 1 or 2
                # if (is.null(initparam)) {
                #         initparam <- vector(mode = "list", length = np)
                #         initparam <- lapply(1:np,
                #                             FUN = function(i) initparam[[i]] <- 0.0)
                #         names(initparam) <- names(argumdist)
                # }

                # order of initparam and par_names(names argumdist) must be the same
                initparam <- initparam[names(argumdist)]

        }


        # List of optimizers
        optimizers <- c("AdadeltaOptimizer", "AdagradDAOptimizer", "AdagradOptimizer", "AdamOptimizer", "GradientDescentOptimizer",
                        "MomentumOptimizer", "RMSPropOptimizer")
        # Error in character for optimizer
        if (!(optimizer %in% optimizers)) {
                stop(paste0("Unidentified optimizer. Select one of the following optimizers: \n",
                            paste0(optimizers, collapse = ", ")))
        }


        # Value for tolerance
        tolerance <- list(parameters = tolerance, loss = tolerance, gradients = tolerance)


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
        if (is.character(xdist)) {
                res <- disableagerdist(x, dist, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np, distnotf, xdist, optimizer)
        } else {
                res <- disableagerestim(x, fdp, arguments, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np, optimizer)
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

        listarguments <- list(InstantaneousFailures = list(lambda = NULL),
                              Weibull = list(shape = NULL, scale = NULL),
                              DoubleExponential = list(loc = NULL, scale = NULL))

        return(listarguments[[dist]])

}
