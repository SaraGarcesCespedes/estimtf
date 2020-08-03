#' @title reg_estimtf function
#'
#' @description Function to compute the Maximum Likelihood Estimators of regression parameters using TensorFlow
#'
#' @author Sara Garcés Céspedes
#'
#' @param ydist an object of class "formula" that specifies the distribution of the response variable. FALTA
#' @param formulas a list containing objects of class "formula". Each element of the list specifies the
#' linear predictor for each of the parameters of the distribution of interest. FALTA
#' @param data an optional data frame containing the variables in the model. If these variables are
#' not found in \code{data}, they ara taken from the environment from which \code{reg_estimtf} is called.
#' @param fixparam a list of the fixed parameters of the distribution of interest. The list must contain the parameters values and names.
#' @param initparam a list with initial values of the regression parameters to be estimated. The list must contain the regression parameters values and names.
#' If you want to use the same initial values for all regression parameters associated with a distributional parameter, you can write the
#' name of the distributional parameter and the value.
#' @param link_function a list with names of parameters to be linked and the corresponding link function name. The link functions available are:
#' \code{log} and \code{logit}.
#' @param optimizer a character indicating the name of the TensorFlow optimizer to be used in the estimation process The default value is \code{'AdamOptimizer'}.
#' @param hyperparameters a list with the hyperparameters values of the TensorFlow optimizer. FALTA DETALLES
#' @param maxiter a positive integer indicating the maximum number of iterations for the optimization algorithm. The default value is \code{10000}.
#' @param eager If \code{TRUE}, the estimation process is performed in the eager execution environment. The default value is \code{TRUE}.

#' @return
#'
#' @details \code{reg_estimtf} computes the log-likelihood function of the distribution specified in
#' \code{ydist} with linear predictors specified in \code{formulas}. Then, it finds the regression parameters
#' that maximizes it using TensorFlow.
#'
#' #' @examples
#' #-------------------------------------------------------------
#' # Estimation of parameters of a linear regression model
#' n <- 1000
#' x <- runif(n = 1000, 0, 10)
#' x1 <- runif(n = n, 0, 6)
#' y <- rnorm(n = n, mean = -2 + 3 * x + 9 * x1, sd = exp(3 + 3 * x))
#' data <- data.frame(y = y, x = x, x1 = x1)
#'
#' formulas <- list(loc.fo = ~ x + x1, scale.fo = ~ x)
#'
#' estimation_1 <- reg_estimtf(ydist = "Normal", formulas = formulas, data = data,
#'                             initparam <- list(loc = 1.0, scale = 1.0),
#'                             link_function <- list(scale = "log"), optimizer = "AdamOptimizer",
#'                             hyperparameters = list(learning_rate = 0.1))
#'
#' estimation_1
#'
#' #-------------------------------------------------------------
#' # Estimation of parameters of a linear regression model with one fixed parameter
#' x <- runif(n = 1000, -3, 3)
#' y <- rnorm(n = 1000, mean = 5 - 2 * x, sigma = 3)
#' data <- data.frame(y = y, x = x)
#'
#' formulas <- list(loc.fo = ~ x)
#' initparam <- list(loc = list(Intercept = 1.0, x = 0.0))
#'
#' estimation_2 <- reg_estimtf(ydist = "Normal", formulas = formulas, data = data,
#'                             fixparam = list(scale = 3), initparam = initparam
#'                             link_function <- list(scale = "log"), optimizer = "AdamOptimizer",
#'                             hyperparameters = list(learning_rate = 0.1))
#'
#' estimation_2
#'
#' @export
reg_estimtf <- function(ydist = y ~ Normal, formulas, data = NULL, fixparam = NULL, initparam = NULL, link_function = NULL,
                        optimizer = "AdamOptimizer", hyperparameters = NULL, maxiter = 10000, eager = TRUE) {

        library(EstimationTools) ; library(RelDists) ; library(tensorflow) ; library(reticulate)
        library(dplyr) ; library(stringr) ; library(ggplot2)

        # Errors in arguments
        # Formulas
        if (is.null(names(formulas))) stop(paste0("You must specify parameters ",
                                                    "formulas with the correct ",
                                                    "notation '.fo'"))

        # Error in character xdist
        if (is.null(ydist)) {
                stop("Distribution of response variable y must be specified \n \n")
        }
        if (!inherits(ydist, "formula")) {
                stop(paste0("'ydist' argument must be ", "a formula specifying ", "the distribution of ",
                                                        "the response variable \n \n"))
        }


        # Defining loss function depending on xdist
        distnotf <- c("Poisson", "FWE", "InstantaneousFailures", "Weibull", "Cauchy",
                      "DoubleExponential", "Geometric", "LogNormal")
        if (!(all.vars(ydist)[2] %in% distnotf)) {
                dist <- eval(parse(text = paste("tf$compat$v1$distributions$", all.vars(ydist)[2], sep = "")))
        } else {
                dist <- all.vars(ydist)[2]
        }

        # List of arguments of TensorFlow functions
        if (all.vars(ydist)[2] %in% distnotf) {
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
        if (all.vars(ydist)[2] %in% distnotf){
                np <- length(argumdist) # number of parameters to be estimated
        } else {
                arg <- sapply(1:length(argumdist),
                              FUN = function(x) names(argumdist)[x] != "validate_args" & names(argumdist)[x] != "allow_nan_stats" & names(argumdist)[x] != "name" & names(argumdist)[x] != "dtype")
                np <- sum(arg)
                argumdist <- argumdist[arg]
        }

        # Names of parameters to be estimated
        par_names <- names(argumdist)

        # Errors in list initparam
        if (!is.null(initparam)) {
                if (length(match(names(initparam), names(argumdist))) == 0) {
                        stop(paste0("Names of parameters included in the 'initparam' list do not match with the arguments of ",
                                    dist, " function."))
                } else if (length(match(names(initparam), names(argumdist))) > np) {
                        stop(paste0("Only include in 'initparam' the parameters that are not fixed"))
                } else if (length(match(names(initparam), names(argumdist))) > 0 & length(match(names(initparam), names(argumdist))) < np) {
                        providedvalues <- match(names(initparam), names(argumdist))
                        namesprovidedvalues <- names(initparam)
                        missingvalues <- argumdist[-providedvalues]
                        initparam <- append(initparam, rep(1.0, length(missingvalues)))
                        names(initparam) <- c(namesprovidedvalues, names(missingvalues))
                }
        }

        # If the user do not provide initial values for the parameters to be estimated, by default the values will be 0
        if (is.null(initparam)) {
                initparam <- vector(mode = "list", length = np)
                initparam <- lapply(1:np, FUN = function(i) initparam[[i]] <- 1.0)
                names(initparam) <- names(argumdist)
        }

        # order of formulas and par_names must be the same
        par_names <- gsub("\\..*$", "", names(formulas))
        # order of initparam and par_names must be the same
        initparam <- initparam[par_names]


        # Errors in link_function
        lfunctions <- c("logit", "log")
        if (!is.null(link_function)) {
                if (length(match(gsub("\\..*","",names(link_function)), names(argumdist))) == 0) {
                        stop(paste0("Names of parameters included in the 'link_function' list do not match with the parameters of the ",
                                    dist, " distribution"))
                } else if (length(match(gsub("\\..*","",names(link_function)), names(argumdist))) > np) {
                        stop(paste0("Only include in 'link_function' the parameters that are not fixed"))
                }
                verifylink <- lapply(1:length(link_function), FUN = function(x) {
                        if (!(link_function[[x]] %in% lfunctions)) {
                                stop(paste0("Unidentified link function Select one of the link functions included in the \n",
                                            " following list: ", paste0(lfunctions, collapse = ", ")))
                        }
                })
        }


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


        # Create the design matrix
        design_matrix <- model.matrix.MLreg(formulas, data, ydist, np, par_names)

        # Estimation process starts

        # With eager execution or disable eager execution
        if (eager == TRUE) {
                res <- eagerreg(data, dist, design_matrix, fixparam, initparam, argumdist, opt, hyperparameters, maxiter, tolerance, np, link_function, ydist)
        } else {
                res <- disableagerreg(data, dist, design_matrix, fixparam, initparam, argumdist, opt, hyperparameters, maxiter, tolerance, np, link_function, ydist)
        }

        return(list(tf = res$final, stderrtf = res$standarderror))
}


#------------------------------------------------------------------------
# Functions to create design matrix (EstimationTools package) -----------
#------------------------------------------------------------------------
matrixes <- function(j, formulas, model_frames){
        do.call(what = "model.matrix",
                args = list(object = as.formula(formulas[[j]]),
                            data = model_frames[[j]]))
}

fos_bind <- function(formula, response){
        paste(response, paste(formula, collapse = " "))
}


model.matrix.MLreg <- function(formulas, data, ydist, np, par_names){

        # Errors in formulas
        if (!any(lapply(formulas, class) == "formula")){
                stop("All elements in argument 'formulas' must be of class formula")
        }

        # Number of formulas (one formula for each parameter)
        nfos <- length(formulas)

        if (nfos != np) stop(paste0("Distribution defined for response ",
                                      "variable has ", npar, " parameters to be estimated. ",
                                      "Each parameter must have its own formula"))

        # Response variable
        if (!inherits(ydist, "formula")) stop(paste0("Expression in 'y_dist' ",
                                                       "must be of class 'formula"))
        if (length(ydist) != 3) stop(paste0("Expression in 'y_dist' ",
                                              "must be a formula of the form ",
                                             "'response ~ distribution'"))

        Y <- all.vars(ydist)[1] #Surv_transform(y_dist = y_dist)

        # Extract the right side of formulas
        formulas_corrector <- stringr::str_extract(as.character(formulas), "~.+")
        formulas_tmp <- as.list(formulas_corrector)
        names(formulas_tmp) <- par_names

        # Variables
        fos_mat_char <- lapply(formulas_tmp, fos_bind, response = Y)
        fos_mat <- lapply(fos_mat_char, as.formula)
        list_mfs <- lapply(fos_mat, model.frame, data = data)
        if ( is.null(data) ){
                data_reg <- as.data.frame(list_mfs)
                var_names <- as.character(unlist(sapply(list_mfs, names)))
                names(data_reg) <- var_names
                data_reg <- as.data.frame(data_reg[,unique(var_names)])
                names(data_reg) <- unique(var_names)
                data <- data_reg
        }
        response <- model.frame(fos_mat[[1]], data = data)[, 1]

        # Formulas for 'model.frame'
        mtrxs <- lapply(X = 1:nfos, FUN = matrixes, formulas = fos_mat,
                        model_frames = list_mfs)

        names(mtrxs) <- names(fos_mat)
        mtrxs$y <- response
        # mtrxs$status <- cens[,2:ncol(cens)]
        mtrxs$data_reg <- data
        return(mtrxs)
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
