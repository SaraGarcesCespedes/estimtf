#' @title estimglm function
#'
#' @description Function to fit Generalized Linear Models using TensorFlow.
#'
#' @author Sara Garces Cespedes
#'
#' @param formula an object of class "formula" that specifies the model to be fitted.
#' @param family a description of the error distribution to be used in the model.
#' @param link_function a string that specifies the model link function.
#' @param data an optional data frame containing the variables in the model. If these variables are
#' not found in \code{data}, they ara taken from the environment from which \code{reg_estimtf} is called.
#' @param initcoeff Optional a list with the initial values for model coefficients. Default value: Zeros.
#' @param optimizer a character indicating the name of the TensorFlow optimizer to be used in the estimation process The default value is \code{'AdamOptimizer'}. The available optimizers are:
#' \code{"AdadeltaOptimizer"}, \code{"AdagradDAOptimizer"}, \code{"AdagradOptimizer"}, \code{"AdamOptimizer"}, \code{"GradientDescentOptimizer"},
#' \code{"MomentumOptimizer"} and \code{"RMSPropOptimizer"}.
#' @param hyperparameters a list with the hyperparameters values of the TensorFlow optimizer. FALTA DETALLES
#' @param maxiter a positive integer indicating the maximum number of iterations for the optimization algorithm. The default value is \code{10000}.
#'
#' @return This function returns the estimates for coefficients of Generalized Linear Models. ESTA MAL
#'
#' @details \code{estimglm} computes the log-likelihood function depending on the provided family. Then, it finds the regression coefficients
#' that maximizes it using TensorFlow. ESTA MAL
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
#' # Fit simple linear regression model (cars dataset)
#' data <- data.frame(dist = cars$dist, speed = cars$speed)
#' formula <- dist ~ speed
#' estimation_1 <- estimglm(formula, family = "Normal", link_function = "identity",
#'                          data = data, initcoeff = list(speed = 1),
#'                          optimizer = "AdamOptimizer", hyperparameters = list(learning_rate = 0.1),
#'                          maxiter = 10000)
#'
#' summary(estimation_1)
#'
#' @export
estimglm <- function(formula, family = "Normal", link_function = "identity", data = NULL, initcoeff = NULL,
                        optimizer = "AdamOptimizer", hyperparameters = NULL, maxiter = 10000) {


        call <- match.call()


        # disable eager
        tensorflow::tf$compat$v1$disable_eager_execution()

        # Errors in arguments

        # Errors in formula
        if (class(formula) != "formula"){
                stop("All elements in argument 'formula' must be of class formula")
        }

        # MENSAJE DE ERROR
        if (length(formula) != 3) stop(paste0("Expression in 'formula' ",
                                            "must be a formula of the form ",
                                            "'response ~ distribution'"))

        # Error in family
        available_families <- c("Normal", "Poisson", "Binomial")

        if (!(family %in% available_families)) {
                stop(paste0("The provided family is not available. The following are the \n",
                            " available families: ", paste0(available_families, collapse = ", ")))
        }



        # Errors in data
        # Check that all variables in formula are included in data
        if (!is.null(data)) {
                if (length(na.omit(match(colnames(data), all.vars(formula)[1]))) < 1) {
                        stop(paste0("Data for response variable ", all.vars(formula)[1], " is missing. \n",
                                    "Please include it in the dataframe provided in the 'data' argument."))
                }
        }


        # Defining loss function depending on xdist
        if (family == "Normal" | family == "Poisson") {
                dist <- eval(parse(text = paste("tfprobability::tfp$distributions$", family, sep = "")))
        } else {
                dist <- family
        }

        # List of arguments of TensorFlow functions
        if (family == "Normal" | family == "Poisson") {
                # Arguments names of tf function
                inspect <- reticulate::import("inspect")
                argumdist <- inspect$signature(dist)
                argumdist <- argumdist$parameters$copy()
        } else {
                argumdist <- arguments(dist)

        }


        # Errors in data
        # Check that all variables in formula are included in data
        if (!is.null(data)) {
                if (length(na.omit(match(colnames(data), all.vars(formula)[1]))) < 1) {
                        stop(paste0("Data for response variable ", all.vars(formula)[1], " is missing. \n",
                                    "Please include it in the dataframe provided in the 'data' argument."))
                }
        }



        # Calculate number of parameters to be estimated. Remove from argumdist the arguments that are not related with parameters
        if (family == "Binomial"){
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
        if (!is.null(initcoeff)) {
                initparam <- list(initcoeff)
                #print(initparam)
                if (family == "Normal") {
                        initparam <- append(initparam, 1.0)
                }
                names(initparam) <- names(argumdist)
                #print(initparam)
        } else {
                # If the user do not provide initial values for the parameters to be estimated, by default the values will be 0
                initparam <- vector(mode = "list", length = np)
                initparam <- lapply(1:np, FUN = function(i) initparam[[i]] <- 0.0)
                names(initparam) <- names(argumdist)
        }

        par_names <- names(argumdist)

        # Errors in link_function
        lfunctions <- c("logit", "log", "inverse", "identity")

        if (!is.null(link_function)) {
                if (!(link_function %in% lfunctions)) {
                        stop(paste0("Unidentified link function. Select one of the link functions included in the \n",
                                    " following list: ", paste0(lfunctions, collapse = ", ")))
                }
        }

        # create formulas and link_function lists
        if (family == "Normal") {
                formulas <- list(loc.fo = formula[-2], scale.fo = ~1)
                link_function <- list(loc = link_function)
        } else if (family == "Poisson") {
                formulas <- list(rate.fo = formula[-2])
                link_function <- list(rate = link_function)
        } else {
                formulas <- list(logits.fo = formula[-2])
                link_function <- list(logits = link_function)
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
        ydist <- formula
        fixparam <- NULL
        design_matrix <- model_matrix_MLglm(formulas, data, ydist, np, par_names)

        # Estimation process starts

        # With disable eager execution
        res <- disableagerglm(data,family, dist, design_matrix, fixparam, initparam, argumdist, opt, hyperparameters, maxiter, tolerance, np, link_function, ydist, optimizer)

        result <- list(tf = res$results, vvoc = res$vcov, stderrtf = res$standarderror, dsgmatrix = design_matrix,
                       outputs = res$outputs, call = call, optimizer = optimizer, distribution = family)
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


model_matrix_MLglm <- function(formulas, data, ydist, np, par_names){


        # Number of formulas (one formula for each parameter)
        nfos <- length(formulas)

        Y <- all.vars(ydist)[1] #Surv_transform(y_dist = y_dist)

        # Extract the right side of formulas
        formulas_corrector <- stringr::str_extract(as.character(formulas), "~.+")
        formulas_tmp <- as.list(formulas_corrector)
        names(formulas_tmp) <- par_names

        # check that all explanatory variables are included in data
        dep_variables <- all.vars(ydist)[-1]

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
        # mtrxs$status <- cens[,2:ncol(cens)]
        mtrxs$data_reg <- data
        return(mtrxs)
}


#------------------------------------------------------------------------
# List of arguments for distributions not included in TF ----------------
#------------------------------------------------------------------------
arguments <- function(dist) {

        listarguments <- list(Logistic = list(logits = NULL))

        return(listarguments[[dist]])

}

