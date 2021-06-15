#' @title mlereg_tf function
#'
#' @description Function to compute the Maximum Likelihood Estimators of regression parameters using TensorFlow.
#'
#' @author Sara Garces Cespedes
#'
#' @param ydist an object of class "formula" that specifies the distribution of the response variable.
#' The available distributions are: \code{Normal}, \code{Poisson}, \code{Binomial}, \code{Weibull}, \code{Exponential}, \code{LogNormal},
#' \code{Beta} and \code{Gamma}. If you want to estimate parameters from a distribution different to the ones mentioned above, you must provide the
#' name of an object of class function that contains its probability mass/density function.
#' @param formulas a list containing objects of class "formula". Each element of the list represents the
#' linear predictor for each of the parameters of the regression model. The linear predictor is specified with
#' the name of the parameter and it must contain an \code{~}. The terms on the right side
#' separated by \code{+}.
#' @param data a data frame containing the response variable and the covariates.
#' @param available_distribution logical. If TRUE, the distribution of the response variable is one of the distributions mentioned above in the package.
#' @param fixparam a list containing the fixed parameters of the model. The parameters values and names must be specified in the list.
#' @param initparam a list with initial values of the regression coefficients to be estimated. The list must contain the regression coefficients values and names.
#' If you want to use the same initial values for all regression coefficients associated with a specific parameter, you can specify the
#' name of the parameter and the value. If NULL the default initial value is zero.
#' @param link_function a list with names of parameters to be linked and the corresponding link function name. The available link functions are:
#' \code{log}, \code{logit}, \code{inverse} and \code{identity}.
#' @param optimizer a character indicating the name of the TensorFlow optimizer to be used in the estimation process. The default value is \code{'AdamOptimizer'}. The available optimizers are:
#' \code{"AdadeltaOptimizer"}, \code{"AdagradDAOptimizer"}, \code{"AdagradOptimizer"}, \code{"AdamOptimizer"}, \code{"GradientDescentOptimizer"},
#' \code{"MomentumOptimizer"} and \code{"RMSPropOptimizer"}.
#' @param hyperparameters a list with the hyperparameters values of the selected TensorFlow optimizer. If the hyperparameters are not specified, their default values
#' will be used in the oprimization process. For more details of the hyperparameters go to this URL:
#' \href{https://www.tensorflow.org/api_docs/python/tf/compat/v1/train}{https://www.tensorflow.org/api_docs/python/tf/compat/v1/train}
#' @param maxiter a positive integer indicating the maximum number of iterations for the optimization algorithm. The default value is \code{10000}.
#' @param tolerance a small positive number. When the difference between the loss value or the parameters values from one iteration to another is lower
#' than this value, the optimization process stops.
#'
#' @return This function returns the estimates, standard errors, Z-score and p-values of significance tests
#' of the regression model coefficients as well as some information of the optimization process like the number of
#' iterations needed for convergence.
#'
#' @details \code{mlereg_tf} computes the log-likelihood function based on the distribution specified in
#' \code{ydist} and linear predictors specified in \code{formulas}. Then, it finds the values of the regression coefficients
#' that maximizes this function using the TensorFlow opimizer specified in \code{optimizer}.
#'
#' @note The \code{summary, print} functions can be used with a \code{mlereg_tf} object.
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
#' @examples
#' #-------------------------------------------------------------
#' # Estimation of parameters of a Poisson regression model
#'
#' # Data frame with response variable and covariates
#' counts <- c(18,17,15,20,10,20,25,13,12)
#' outcome <- gl(3,1,9)
#' treatment <- gl(3,3)
#' data <- data.frame(treatment, outcome, counts)
#'
#' # Define the linear predictors for each parameter
#' formulas <- list(lambda = ~ outcome + treatment)
#'
#' # Use the mlereg_tf function
#' estimation_1 <- mlereg_tf(ydist =  counts ~ Poisson,
#'                             formulas = formulas,
#'                             data = data,
#'                             initparam = list(lambda = 1.0),
#'                             optimizer = "AdamOptimizer",
#'                             link_function = list(lambda = "log"),
#'                             hyperparameters = list(learning_rate = 0.1))
#'
#' # Get the summary of estimates
#' summary(estimation_1)
#'
#' #-------------------------------------------------------------
#' # Estimation of parameters of a linear regression model with one fixed parameter
#'
#' # Data frame with response variable and covariates
#' x <- runif(n = 1000, -3, 3)
#' y <- rnorm(n = 1000, mean = 5 - 2 * x, sd = 3)
#' data <- data.frame(y = y, x = x)
#'
#' # Define the linear predictors for each parameter and the initial values
#' formulas <- list(mean = ~ x)
#' initparam <- list(mean = list(Intercept = 1.0, x = 0.0))
#'
#' # Use the mlereg_tf function
#' estimation_2 <- mlereg_tf(ydist = y ~ Normal,
#'                             formulas = formulas,
#'                             data = data,
#'                             fixparam = list(sd = 3),
#'                             initparam = initparam,
#'                             optimizer = "AdamOptimizer",
#'                             hyperparameters = list(learning_rate = 0.1))
#'
#' # Get the summary of estimates
#' summary(estimation_2)
#'
#' #-------------------------------------------------------------
#' # Estimation of parameters from Instantaneous Failures distribution
#'
#' # Create an R function that represents the probability density function
#' pdf <- function(y, lambda) { (1 / ((lambda ^ 2) * (lambda - 1))) *
#'                              (lambda^2 + y - 2*lambda) * exp(-y/lambda) }
#'
#' # Data frame with response variable and covariates
#' y <-  c(3.4, 0.0, 0.0, 15.8, 232.8, 8.8, 123.2, 47, 154, 103.2, 89.8,  12.2)
#' data <- data.frame(y)
#'
#' # Use the mlereg_tf function
#' estimation_3 <- mlereg_tf(ydist = y ~ pdf,
#'                              formulas = list(lambda = ~1),
#'                              data = data,
#'                              initparam = list(lambda = rnorm(1, 5, 1)),
#'                              available_distribution = FALSE,
#'                              optimizer = "AdamOptimizer",
#'                              hyperparameters = list(learning_rate = 0.1),
#'                              maxiter = 10000)
#'
#' # Get the summary of estimates
#' summary(estimation_3)
#'
#'
#' @export
mlereg_tf <- function(ydist = y ~ Normal, formulas, data, available_distribution = TRUE, fixparam = NULL, initparam = NULL, link_function = NULL,
                      optimizer = "AdamOptimizer", hyperparameters = NULL, maxiter = 10000, tolerance = .Machine$double.eps) {


        call <- match.call()

        # Errors in arguments

        # missing arguments
        if (missing(formulas)) {
                stop(paste0("Argument 'formulas' is missing, with no default"))
        } else if (missing(data)) {
                stop(paste0("Argument 'data' is missing, with no default"))
        } else if (is.null(data)) {
                stop(paste0("Please provide a data frame with the response variable and the covariates"))
        }

        # Formulas
        if (is.null(names(formulas))) stop(paste0("You must specify parameters ",
                                                  "names in the argument formulas"))

        # Errors in formulas
        if (!any(lapply(formulas, class) == "formula")){
                stop("All elements in argument 'formulas' must be of class formula")
        }

        # Number of formulas (one formula for each parameter)
        n_formulas <- length(formulas)


        # Response variable
        if (!inherits(ydist, "formula")) stop(paste0("Expression in 'y_dist' ",
                                                     "must be of class 'formula"))
        if (length(ydist) != 3) stop(paste0("Expression in 'y_dist' ",
                                            "must be a formula of the form ",
                                            "'response ~ distribution'"))


        # Error in character Ydist
        if (is.null(ydist)) {
                stop("Distribution of response variable y must be specified \n \n")
        }

        if (!inherits(ydist, "formula")) {
                stop(paste0("'ydist' must be a formula specifying the distribution of ",
                            "the response variable \n \n"))
        }


        if (available_distribution == TRUE) {

                # Import tensorflow_probability module
                tf_prob <- reticulate::import("tensorflow_probability")

                # change names of parameters to match TF parameters
                names_param <- names(formulas)
                #names_param_final <- stringr::str_remove_all(names_param, ".fo")
                names_new <- vector(mode = "numeric", length = length(names_param))
                names_new <- sapply(1:length(names_param), FUN = function(i) names_new[i] <- paste0(parameter_name_tf(names_param[i], all.vars(ydist)[2])))
                names(formulas) <- names_new




                # Defining loss function depending on xdist
                distdisponibles <- c("Normal", "Poisson", "Gamma", "LogNormal", "Weibull", "Exponential",
                                     "Beta", "Binomial")
                distnotf <- c("FWE", "InstantaneousFailures", "DoubleExponential")

                if (!(all.vars(ydist)[2] %in% distdisponibles)) {
                        stop(paste0("The distribution is not available. The following are the \n",
                                    " available distributions: ", paste0(distdisponibles, collapse = ", ")))
                }

                if (!(all.vars(ydist)[2] %in% distnotf)) {
                        #dist <- eval(parse(text = paste("tfprobability::tfp$distributions$", all.vars(ydist)[2], sep = "")))
                        dist <- eval(parse(text = paste("tf_prob$distributions$", all.vars(ydist)[2], sep = "")))
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
                                            "Please include it in the data frame provided in the 'data' argument."))
                        }
                }

                # Errors in list fixparam
                # Update argumdist. Leaves all the arguments of the TF distribution except the ones that are fixed
                if (!is.null(fixparam)) {

                        # change names of parameters to match TF parameters
                        names_param <- names(fixparam)
                        names_new <- vector(mode = "numeric", length = length(names_param))
                        names_new <- sapply(1:length(names_param), FUN = function(i) names_new[i] <- parameter_name_tf(names_param[i], all.vars(ydist)[2]))
                        names(fixparam) <- names_new

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

                        if (!inherits(initparam, "list")) {
                                stop(paste0("'initparam' must be a nonempty list containing the initial values, \n",
                                            "of the parameters to be estimated. \n \n"))
                        }

                        # change names of parameters to match TF parameters
                        names_param <- names(initparam)
                        names_new <- vector(mode = "numeric", length = length(names_param))
                        names_new <- sapply(1:length(names_param), FUN = function(i) names_new[i] <- parameter_name_tf(names_param[i], all.vars(ydist)[2]))
                        names(initparam) <- names_new

                        if (all(names(initparam) %in% names(argumdist)) == FALSE) {
                                stop(paste0("Some or all of the parameters included in the 'initparam' list do not match with the arguments of ",
                                            all.vars(ydist)[2], " distribution."))
                        } else if (length(na.omit(match(names(initparam), names(argumdist)))) > np) {
                                stop(paste0("Only include in 'initparam' the parameters that are not fixed"))
                        } else {

                                providedvalues <- match(names(initparam), names(argumdist))
                                namesprovidedvalues <- names(initparam)
                                missingvalues <- argumdist[-providedvalues]
                                initparam <- append(initparam, rep(0.0, length(missingvalues)))
                                names(initparam) <- c(namesprovidedvalues, names(missingvalues))
                        }
                }

                # # If the user do not provide initial values for the parameters to be estimated, by default the values will be 0
                if (is.null(initparam)) {
                        initparam <- vector(mode = "list", length = np)
                        initparam <- lapply(1:np, FUN = function(i) initparam[[i]] <- 0.0)
                        names(initparam) <- names(argumdist)
                }



                # order of formulas and par_names must be the same
                par_names <- gsub("\\..*$", "", names(formulas))
                # order of initparam and par_names must be the same
                initparam <- initparam[par_names]

        } else {

                # List of arguments FDP
                function_loss <- all.vars(ydist)[2]

                # respone variable
                response_var <- all.vars(ydist)[1]

                if (function_loss %in% ls(envir = .GlobalEnv)) {
                        fdp <- get(function_loss, envir = .GlobalEnv)
                } else {
                        stop(paste0("Function '", function_loss, "' not found in Global Environment."))
                }

                arguments <- formals(fdp)
                arguments <- as.list(arguments)

                # eliminar variable respuesta de lista vartotal
                if (response_var %in% names(arguments)) {
                        arguments[[response_var]] <- NULL
                } else {
                        message('Caught an error!')
                        stop(paste0("Argument '", response_var, "' is missing in the probability mass/density function."))
                }

                # remove fixed parameters
                # arguments_fixed <- vector(mode = "list", length = length(arguments))
                # arguments_fixed <- lapply(1:length(arguments), FUN = function(i) arguments_fixed[[i]] <- ifelse(is.numeric(arguments[[i]]), arguments[[i]], NA))
                # names(arguments_fixed) <- names(arguments)
                # fixparam <- arguments_fixed %>% purrr::discard(is.na)

                # # Calculate number of parameters to be estimated
                # argumdist <- arguments_fixed %>% purrr::discard(is.numeric)
                # np <- length(argumdist)


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

                        if (!inherits(initparam, "list")) {
                                stop(paste0("'initparam' must be a nonempty list containing the initial values, \n",
                                            "of the parameters to be estimated. \n \n"))
                        }

                        if (all(names(initparam) %in% names(argumdist)) == FALSE) {
                                stop(paste0("Some or all of the parameters included in the 'initparam' list do not match with the arguments of ",
                                            all.vars(ydist)[2], " distribution."))
                        } else if (length(na.omit(match(names(initparam), names(argumdist)))) > np) {
                                stop(paste0("Only include in 'initparam' the parameters that are not fixed"))
                        } else {

                                providedvalues <- match(names(initparam), names(argumdist))
                                namesprovidedvalues <- names(initparam)
                                missingvalues <- argumdist[-providedvalues]
                                initparam <- append(initparam, rep(0.0, length(missingvalues)))
                                names(initparam) <- c(namesprovidedvalues, names(missingvalues))
                        }
                }

                # If the user do not provide initial values for the parameters to be estimated, by default the values will be 0
                if (is.null(initparam)) {
                        initparam <- vector(mode = "list", length = np)
                        initparam <- lapply(1:np, FUN = function(i) initparam[[i]] <- 0.0)
                        names(initparam) <- names(argumdist)
                }

                # order of formulas and par_names must be the same
                par_names <- gsub("\\..*$", "", names(formulas))
                # order of initparam and par_names must be the same
                initparam <- initparam[par_names]
        }



        # Errors in link_function
        lfunctions <- c("logit", "log", "inverse", "identity")
        if (!is.null(link_function)) {

                verifylink <- lapply(1:length(link_function), FUN = function(x) {
                        if (!(link_function[[x]] %in% lfunctions)) {
                                stop(paste0("Unidentified link function. Select one of the link functions included in the \n",
                                            " following list: ", paste0(lfunctions, collapse = ", ")))
                        }
                })


                # change names of parameters to match TF parameters
                if (available_distribution == TRUE) {
                        names_param <- names(link_function)
                        names_new <- vector(mode = "numeric", length = length(names_param))
                        names_new <- sapply(1:length(names_param), FUN = function(i) names_new[i] <- parameter_name_tf(names_param[i], all.vars(ydist)[2]))
                        names(link_function) <- names_new
                }

                if (all(names(link_function) %in% names(argumdist)) == FALSE) {
                        stop(paste0("Names of parameters included in the 'link_function' list do not match with the parameters of the ",
                                    all.vars(ydist)[2], " distribution"))
                } else if (length(na.omit(match(names(link_function), names(argumdist)))) > np) {
                        stop(paste0("Only include in 'link_function' the parameters that are not fixed"))
                }

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
        design_matrix <- model_matrix_MLreg(formulas, data, ydist, np, par_names, n_formulas)

        # Estimation process starts

        # With disable eager execution
        if (available_distribution == TRUE) {
                res <- disableagerreg(data, dist, design_matrix, fixparam, initparam, argumdist, opt, hyperparameters, maxiter, tolerance, np, link_function, ydist, distnotf, optimizer)
        } else {
                res <- disableagerregpdf(data, fdp, design_matrix, fixparam, initparam, argumdist, opt, hyperparameters, maxiter, tolerance, np, link_function, ydist, distnotf, optimizer, arguments, response_var)
        }

        result <- list(tf = res$results, vvoc = res$vcov, stderrtf = res$standarderror, dsgmatrix = design_matrix,
                       outputs = res$outputs, call = call, optimizer = optimizer, distribution = all.vars(ydist)[2])
        class(result) <- "MLEtf"
        return(result)
}


#------------------------------------------------------------------------
# Function to create design matrix --------------------------------------
#------------------------------------------------------------------------
model_matrix_MLreg <- function(formulas, data, ydist, np, par_names, n_formulas){

        # Error
        if (n_formulas != np) stop(paste0("Distribution defined for response ",
                                          "variable has ", np, " parameters to be estimated. ",
                                          "Each parameter must have its own formula"))

        # name and data for response variable
        Y <- all.vars(ydist)[1]

        tryCatch(
                expr = {
                        response_data <- data[, Y]
                },
                error = function(e){
                        #message('Caught an error!')
                        #print(e)
                        stop(paste0("Data for response variable '", Y, "' is missing in the provided dataframe."))
                }
        )

        # Extract the right side of formulas
        formulas_right <- stringr::str_extract(as.character(formulas), "~.+")
        formulas_final <- as.list(formulas_right)
        names(formulas_final) <- par_names

        # check that all explanatory variables are included in data
        split_form <- sapply(1:length(formulas_final), FUN = function(x) formulas_final[[x]] %>% stringr::str_remove("\\~"))
        split_form <- sapply(1:length(split_form), FUN = function(x) split_form[[x]] %>% stringr::str_split("\\+"))

        dep_variables <- unlist(split_form)
        dep_variables <- sapply(1:length(dep_variables), FUN = function(x) dep_variables[x] = trimws(dep_variables[x]))
        dep_variables <- dep_variables[!dep_variables == 1]
        dep_variables <- dep_variables[!dep_variables == -1]
        dep_variables <- unique(dep_variables)

        if ((ncol(data) - 1) < length(dep_variables)) {
                stop(paste0("Data for some explanatory variables is missing. Please include it\n",
                            "in the data frame provided in the 'data' argument."))
        }

        # model frame for each parameter
        formulas_modelframe <- lapply(X = 1:n_formulas, FUN = function(x) as.formula(paste(Y, paste(formulas_final[[x]], collapse = " "))))
        names(formulas_modelframe) <- par_names
        list_modelframe <- lapply(formulas_modelframe, model.frame, data = data)

        # model matrix for each ´parameter
        design_matrix <- lapply(X = 1:n_formulas, FUN = function(x) model.matrix(formulas_modelframe[[x]], list_modelframe[[x]]))

        names(design_matrix) <- names(formulas_modelframe)
        design_matrix$y <- response_data
        design_matrix$data_reg <- data

        return(design_matrix)
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
