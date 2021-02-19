#' @title reg_estimtf function
#'
#' @description Function to compute the Maximum Likelihood Estimators of regression parameters with linear predictors using TensorFlow.
#'
#' @author Sara Garces Cespedes
#'
#' @param ydist an object of class "formula" that specifies the distribution of the response variable.
#' The available distributions are: The available distributions are: \code{Normal}, \code{Poisson}, \code{Binomial}, \code{Weibull}, \code{Exponential}, \code{LogNormal}, \code{Beta} and \code{Gamma}.
#' @param formulas a list containing objects of class "formula". Each element of the list represents the
#' linear predictor for each of the parameters of the regression model. The linear predictor is specified with
#' the name of the parameter followed by \code{.fo} and it must contain an \code{~} and the terms on the right side
#' separated by \code{+}.
#' @param data a data frame containing the variables in the regression model.
#' @param fixparam a list containing the fixed parameters of the model. The parameters values and names must be specified in the list.
#' @param initparam a list with initial values of the regression coefficients to be estimated. The list must contain the regression coefficients values and names.
#' If you want to use the same initial values for all regression coefficients associated with a specific parameter, you can specify the
#' name of the parameter and the value.
#' @param link_function a list with names of parameters to be linked and the corresponding link function name. The available link functions are:
#' \code{log}, \code{logit}, \code{inverse} and \code{identity}.
#' @param optimizer a character indicating the name of the TensorFlow optimizer to be used in the estimation process. The default value is \code{'AdamOptimizer'}. The available optimizers are:
#' \code{"AdadeltaOptimizer"}, \code{"AdagradDAOptimizer"}, \code{"AdagradOptimizer"}, \code{"AdamOptimizer"}, \code{"GradientDescentOptimizer"},
#' \code{"MomentumOptimizer"} and \code{"RMSPropOptimizer"}.
#' @param hyperparameters a list with the hyperparameters values of the TensorFlow optimizer. (See URL for details of hyperparameters.)
#' @param maxiter a positive integer indicating the maximum number of iterations for the optimization algorithm. The default value is \code{10000}.
#'
#' @return This function returns the estimates, standard errors, t values and p-values of significance tests
#' for the regression model coefficients as well as some information of the optimization process like the number of
#' iterations needed for convergence.
#'
#' @details \code{reg_estimtf} computes the log-likelihood function based on the distribution specified in
#' \code{ydist} and linear predictors specified in \code{formulas}. Then, it finds the regression coefficients
#' that maximizes it using TensorFlow.
#'
#' @note The \code{summary, print} functions can be used with a \code{reg_estimtf} object.
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
#' @importFrom fastDummies dummy_cols
#' @import tensorflow
#' @import tfprobability
#'
#' @examples
#' #-------------------------------------------------------------
#' # Estimation of parameters of a Poisson regression model
#' counts <- c(18,17,15,20,10,20,25,13,12)
#' outcome <- gl(3,1,9)
#' treatment <- gl(3,3)
#' data <- data.frame(treatment, outcome, counts)
#'
#' formulas <- list(rate.fo = ~ outcome + treatment)
#' estimation_1 <- reg_estimtf(ydist =  counts ~ Poisson, formulas = formulas, data = data,
#'                             initparam = list(rate = 1.0), optimizer = "AdamOptimizer",
#'                             link_function = list(rate = "log"),
#'                             hyperparameters = list(learning_rate = 0.1))
#' summary(estimation_1)
#'
#' #-------------------------------------------------------------
#' # Estimation of parameters of a linear regression model with one fixed parameter
#' x <- runif(n = 1000, -3, 3)
#' y <- rnorm(n = 1000, mean = 5 - 2 * x, sd = 3)
#' data <- data.frame(y = y, x = x)
#'
#' formulas <- list(loc.fo = ~ x)
#' initparam <- list(loc = list(Intercept = 1.0, x = 0.0))
#'
#' estimation_2 <- reg_estimtf(ydist = y ~ Normal, formulas = formulas, data = data,
#'                             fixparam = list(scale = 3), initparam = initparam,
#'                             optimizer = "AdamOptimizer",
#'                             hyperparameters = list(learning_rate = 0.1))
#'
#' summary(estimation_2)
#'
#' #-------------------------------------------------------------
#' # Estimation of parameters using cars dataset
#' data <- data.frame(dist = cars$dist, speed = cars$speed)
#' formulas <- list(loc.fo = ~ speed, scale.fo = ~ 1)
#'
#' estimation_3 <- reg_estimtf(ydist = dist ~ Normal, formulas = formulas, data = data,
#'                             initparam = list(loc = 1.0, scale = 1.0), optimizer = "AdamOptimizer",
#'                             hyperparameters = list(learning_rate = 0.1))
#'
#' summary(estimation_3)
#'
#' @export
reg_estimtf <- function(ydist = y ~ Normal, formulas, data = NULL, fixparam = NULL, initparam = NULL, link_function = NULL,
                        optimizer = "AdamOptimizer", hyperparameters = NULL, maxiter = 10000) {


        call <- match.call()

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
                stop(paste0("'ydist' argument must be a formula specifying the distribution of ",
                                                        "the response variable \n \n"))
        }


        # Defining loss function depending on xdist
        distdisponibles <- c("Normal", "Poisson", "Gamma", "LogNormal", "Weibull", "Exponential",
                             "Beta", "Binomial")
        distnotf <- c("FWE", "InstantaneousFailures", "DoubleExponential")

        if (!(all.vars(ydist)[2] %in% distdisponibles)) {
                stop(paste0("The distribution is not available. The following are the \n",
                            " available distributions: ", paste0(distdisponibles, collapse = ", ")))
        }

        if (!(all.vars(ydist)[2] %in% distnotf)) {
                dist <- eval(parse(text = paste("tfprobability::tfp$distributions$", all.vars(ydist)[2], sep = "")))
        } else {
                dist <- all.vars(ydist)[2]
        }

        # List of arguments of TensorFlow functions
        if (all.vars(ydist)[2] %in% distnotf) {
                argumdist <- arguments(dist)
        } else {
                # Arguments names of tf function
                inspect <- reticulate::import("inspect")
                argumdist <- inspect$signature(dist)
                argumdist <- argumdist$parameters$copy()
        }


        # Errors in data
        # Check that all variables in formula are included in data
        if (!is.null(data)) {
                if (length(na.omit(match(colnames(data), all.vars(ydist)[1]))) < 1) {
                        stop(paste0("Data for response variable ", all.vars(ydist)[1], " is missing. \n",
                                    "Please include it in the dataframe provided in the 'data' argument."))
                }
        }

        # Errors in list fixparam
        # Update argumdist. Leaves all the arguments of the TF distribution except the ones that are fixed
        if (!is.null(fixparam)) {
                if (length(na.omit(match(names(argumdist), names(fixparam)))) == 0) {
                        stop(paste0("Names of fixed parameters do not match with the arguments of \n",
                                    all.vars(ydist)[2], " distribution"))
                } else if (length(na.omit(match(names(fixparam), names(argumdist)))) > 0) {
                        fixed <- match(names(fixparam), names(argumdist))
                        argumdist <- argumdist[-fixed]
                }
        }


        # Calculate number of parameters to be estimated. Remove from argumdist the arguments that are not related with parameters
        if (all.vars(ydist)[2] %in% distnotf){
                np <- length(argumdist) # number of parameters to be estimated
        } else {
                arg <- sapply(1:length(argumdist),
                              FUN = function(x) names(argumdist)[x] != "validate_args" & names(argumdist)[x] != "allow_nan_stats" &
                                      names(argumdist)[x] != "name" & names(argumdist)[x] != "dtype" &
                                      names(argumdist)[x] != "interpolate_nondiscrete" & names(argumdist)[x] != "log_rate"&
                                      names(argumdist)[x] != "force_probs_to_zero_outside_support" & names(argumdist)[x] != "logits")
                np <- sum(arg)
                argumdist <- argumdist[arg]
        }

        # Errors in list initparam
        if (!is.null(initparam)) {
                if (all(names(initparam) %in% names(argumdist)) == FALSE) {
                        stop(paste0("Some or all of the parameters included in the 'initparam' list do not match with the arguments of ",
                                    all.vars(ydist)[2], " distribution."))
                } else if (length(na.omit(match(names(initparam), names(argumdist)))) > np) {
                        stop(paste0("Only include in 'initparam' the parameters that are not fixed"))
                } else {

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
        lfunctions <- c("logit", "log", "inverse", "identity")
        if (!is.null(link_function)) {
                if (all(names(link_function) %in% names(argumdist)) == FALSE) {
                        stop(paste0("Names of parameters included in the 'link_function' list do not match with the parameters of the ",
                                    all.vars(ydist)[2], " distribution"))
                } else if (length(na.omit(match(names(link_function), names(argumdist)))) > np) {
                        stop(paste0("Only include in 'link_function' the parameters that are not fixed"))
                }
                verifylink <- lapply(1:length(link_function), FUN = function(x) {
                        if (!(link_function[[x]] %in% lfunctions)) {
                                stop(paste0("Unidentified link function. Select one of the link functions included in the \n",
                                            " following list: ", paste0(lfunctions, collapse = ", ")))
                        }
                })
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
                        stop(paste0("The hyperparameters provided do not match with the hyperparameters of ","TensorFlow ", optimizer, "."))
                }
        } else if (is.null(hyperparameters)) {
                hyperparameters <- vector(mode = "list", length = length(argumopt))
                names(hyperparameters) <- names(argumopt)
                splitarg <- sapply(1:length(argumopt), FUN = function(x) argumopt[[x]] %>% stringr::str_split("\\="))
                hyperparameters <- lapply(1:length(hyperparameters),
                                          FUN = function(i) hyperparameters[[i]] <- ifelse(splitarg[[i]][2] == "True" | splitarg[[i]][2] == "False",
                                                                                           splitarg[[i]][2], as.numeric(splitarg[[i]][2])))
        }


        # Create the design matrix
        design_matrix <- model_matrix_MLreg(formulas, data, ydist, np, par_names)

        # Estimation process starts

        # With disable eager execution
        res <- disableagerreg(data, dist, design_matrix, fixparam, initparam, argumdist, opt, hyperparameters, maxiter, tolerance, np, link_function, ydist, distnotf, optimizer)

        result <- list(tf = res$results, vvoc = res$vcov, stderrtf = res$standarderror, dsgmatrix = design_matrix,
                       outputs = res$outputs, call = call, optimizer = optimizer, distribution = all.vars(ydist)[2])
        class(result) <- "MLEtf"
        return(result)
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


model_matrix_MLreg <- function(formulas, data, ydist, np, par_names){

        # Errors in formulas
        if (!any(lapply(formulas, class) == "formula")){
                stop("All elements in argument 'formulas' must be of class formula")
        }

        # Number of formulas (one formula for each parameter)
        nfos <- length(formulas)

        if (nfos != np) stop(paste0("Distribution defined for response ",
                                      "variable has ", np, " parameters to be estimated. ",
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

        # check that all explanatory variables are included in data
        split_form <- sapply(1:length(formulas_tmp), FUN = function(x) formulas_tmp[[x]] %>% stringr::str_remove("\\~"))
        split_form <- sapply(1:length(split_form), FUN = function(x) split_form[[x]] %>% stringr::str_split("\\+"))

        dep_variables <- unlist(split_form)
        dep_variables <- sapply(1:length(dep_variables), FUN = function(x) dep_variables[x] = trimws(dep_variables[x]))
        dep_variables <- dep_variables[!dep_variables == 1]
        dep_variables <- dep_variables[!dep_variables == -1]
        dep_variables <- unique(dep_variables)

        if ((ncol(data) - 1) < length(dep_variables)) {
                stop(paste0("Data for some explanatory variables is missing. Please include it\n",
                            "in the dataframe provided in the 'data' argument."))
        }

        # Variables
        fos_mat_char <- lapply(formulas_tmp, fos_bind, response = Y)
        fos_mat <- lapply(fos_mat_char, as.formula)
        list_mfs <- lapply(fos_mat, model.frame, data = data)

        if (is.null(data)){
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
        mtrxs$data_reg <- data

        return(mtrxs)
}

#------------------------------------------------------------------------
# List of arguments for distributions not included in TF ----------------
#------------------------------------------------------------------------
arguments <- function(dist) {

        listarguments <- list(InstantaneousFailures = list(lambda = NULL),
                              #Weibull = list(shape = NULL, scale = NULL),
                              DoubleExponential = list(loc = NULL, scale = NULL),
                              Binomial = list(logits = NULL))

        return(listarguments[[dist]])

}
