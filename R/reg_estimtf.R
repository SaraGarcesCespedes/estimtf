#' @title reg_estimtf function
#'
#' @description Function to compute the Maximum Likelihood Estimators of regression parameters using TensorFlow
#'
#' @author Sara Garcés Céspedes
#'
#' @param formula
#' @param ydist
#' @param data
#' @param subset
#' @param fixparam
#' @param initparam
#' @param link_function
#' @param optimizer
#' @param hyperparameters
#' @param maxiter
#' @param tolerance
#' @param eager
#' @param comparison
#' @param lower
#' @param upper
#' @param method
#'
#' @return
#' @export
#'
#' @examples
reg_estimtf <- function(formula, ydist = "Normal", data = NULL, subset = NULL, fixparam = NULL, initparam = NULL, link_function = NULL,
                        optimizer = "AdamOptimizer", hyperparameters = NULL, maxiter = 1000, tolerance = NULL, eager = TRUE, comparison = FALSE,
                        lower = NULL, upper = NULL, method = "nlminb") {


        # Errors in arguments

        # Error in formula
        # Error in matrix data

        # Error in character xdist
        if (is.null(ydist)) {
                stop("Distribution of response variable y must be specified \n \n")
        }
        if (!is.character(ydist)) {
                stop("'ydist' must be a character \n \n")
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
                        #np <- length(argumdist)
                }
        }


        # Calculate number of parameters to be estimated. Remove from argumdist the arguments that are not related with parameters
        if (dist == "Instantaneous Failures" | dist == "Poisson" | dist == "FWE"){
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
                        stop(paste0("Names of parameters included in the 'initparam' list do not match with the arguments of ",
                                    dist, " function."))
                } else if (length(match(names(initparam), names(argumdist))) > np) {
                        stop(paste0("Only include in 'initparam' the names of parameters that are not fixed"))
                }
        }

        # Errors in link_function
        if (!is.null(link_function)) {
                if (length(match(names(link_function), names(argumdist))) == 0) {
                        stop(paste0("Names of parameters included in the 'link_function' list do not match with the parameters of the ",
                                    dist, " distribution"))
                } else if (length(match(names(link_function), names(argumdist))) > np) {
                        stop(paste0("Only include in 'link_function' the names of parameters that are not fixed"))
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


        # If the user do not provide initial values for the parameters to be estimated, by default the values will be 0
        if (is.null(initparam)) {
                initparam <- vector(mode = "list", length = np)
                if (dist == "Instantaneous Failures" | dist == "Poisson" | dist == "FWE"){
                        param <- names(argumdist)
                } else {
                        param <- names(argumdist)[which(names(argumdist)[x] != "validate_args" & names(argumdist)[x] != "allow_nan_stats" & names(argumdist)[x] != "name" & names(argumdist)[x] != "dtype")]
                }
                names(initparam) <- c(param)
                for (i in 1:length(np)) initparam[[i]] <- 0.0 #SEGURAMENTE SE PUEDE HACER MAS EFICIENTE
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
                res <- eagerreg(x, dist, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np)
        } else {
                res <- disableagerreg(x, dist, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np)
        }














}
