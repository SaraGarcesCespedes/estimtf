#' @title glm_estimtf function
#'
#' @description Function to compute the Maximum Likelihood Estimators of regression parameters from Generalized
#' Linear Models using TensorFlow.
#'
#' @author Sara Garces Cespedes
#'
#' @param formula an object of class "formula" that specifies the model to be fitted.
#' @param family a description of the error distribution to be used in the model.
#' @param link_function a string that specifies the model link function.
#' @param data an optional data frame containing the variables in the model. If these variables are
#' not found in \code{data}, they ara taken from the environment from which \code{reg_estimtf} is called.
#' @param initcoeff Optional a list with the initial values for model coefficients. Default value: Zeros.
#' @param maxiter a positive integer indicating the maximum number of iterations for the optimization algorithm. The default value is \code{10000}.
#'
#' @return This function returns the estimates for parameters of Generalized Linear Models. These parameters are
#' estimated using the Fisher scoring method.
#'
#' @details \code{glm_estimtf} uses the Fisher scoring method to determine the maximum of the likelihood function.
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
#' @examples
#'
#' ctl <- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
#' trt <- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
#' group <- gl(2, 10, 20, labels = c("Ctl","Trt"))
#' weight <- c(ctl, trt)
#'
#' family <- "Normal"
#' link_function <- "identity"
#' formula <- weight ~ group
#' data <- data.frame(weight = weight, group = group)
#' estimation <- glm_estimtf(formula = formula, family = family,
#'                           link_function = link_function, data = data)
#'
#' print(estimation)
#'
#' @export
glm_estimtf <- function(formula, family = "Normal", link_function = "identity", data = NULL, initcoeff = NULL, maxiter = 10000) {

        #suppressMessages(library(EstimationTools)) ; suppressMessages(library(RelDists)) ;
        #library(tensorflow)
        #library(reticulate)
        #library(dplyr)
        #library(stringr)
        #library(ggplot2)
        #library(tfprobability)
        #library(purrr)

        call <- match.call()

        # disable eager
        tensorflow::tf$compat$v1$disable_eager_execution()

        # Errors in arguments

        # Errors in formula
        if (class(formula) != "formula"){
                stop("All elements in argument 'formula' must be of class formula")
        }

        # Error in family
        available_families <- c("Bernoulli",  "Binomial", "Gamma", "LogNormal",  "NegativeBinomial",
                                "NHEormal", "Poisson", "Normal")

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

        # Build design matrix
        design_matrix <- model_matrix_GLM(formula, data)
        design_matrix_final <- design_matrix$design_matrix
        y_data <- design_matrix$y
        x_data <- design_matrix$design_matrix
        n_coeff <- ncol(design_matrix_final) # number of coefficients

        # order of initcoeff must be the same as order of design matrix columns
        colnames(design_matrix_final) <- stringr::str_remove_all(colnames(design_matrix_final), "[\\(\\)]") # remove parenthesis of Intercept
        coeff_names <- colnames(design_matrix_final)

        # Errors in list initparam
        if (!is.null(initcoeff)) {
                len_initcoeff <- length(initcoeff)
                if (len_initcoeff < n_coeff) {
                        stop(paste0("There are missing values in the 'initcoeff' list. Please add initial values for all the \n",
                                    "coefficients or use the default values."))
                } else if (len_initcoeff > n_coeff) {
                        stop(paste0("The list with initial values for the coefficients have more values than expected. \n",
                                    "Please check the 'initcoeff' list."))
                } else {
                        initcoeff <- initcoeff[coeff_names]
                        init_values_coeff <- unlist(initcoeff)
                }
        }

        # define X and Y
        X <- tensorflow::tf$constant(x_data, dtype=tf$float32)
        Y <- tensorflow::tf$constant(y_data, dtype=tf$float32)

        family_new <- tf_family(family, link_function)

        if (family_new == "NotAvailable") {
                stop(paste0("The family or link_function are not available. Please check the documentation",
                            " to see what combinations of family and link function are available in this package."))
        } else {
                family_new <- family_new
        }

        # Estimation
        if (!is.null(initcoeff)) {
                estimation <- tfprobability::glm_fit(X, Y, model = family_new, model_coefficients_start = init_values_coeff)
        } else {
                estimation <- tfprobability::glm_fit(X, Y, model = family_new)
        }

        sess <- tf$compat$v1$Session()

        coeff_values <- as.numeric(sess$run(estimation[[1]]))
        linear_response <- as.numeric(sess$run(estimation[[2]]))
        is_converged <- sess$run(estimation[[3]])
        num_iterations <- sess$run(estimation[[4]])

        #list_coeff <- vector(mode = "list", length = length(coeff_values))
        #list_coeff <- lapply(1:length(coeff_values), FUN = function(i) list_coeff[[i]] <- coeff_values[i])
        #names(list_coeff) <- names(coeff_names)

        terms_formula <- terms(formula)

        dep_variable <- all.vars(formula)[1]
        data_ind_variables <- data %>% select(-all_of(dep_variable))
        num_ind_variables <- length(all.vars(formula)) - 1
        levels_variables <- vector(mode = "list", length = num_ind_variables)
        levels_variables <- lapply(1:num_ind_variables, FUN = function(x) levels_variables[[x]] = levels(data_ind_variables[,1]))
        names(levels_variables) <- colnames(data_ind_variables)
        #print("levels")
        #print(levels_variables)

        outputs <- list(type = "MLEglm", formula = formula, tt = terms_formula, xlevels = levels_variables)
        result <- list(outputs = outputs, coeff_estim = coeff_values, names_coeff = coeff_names, converged = is_converged, iterations = num_iterations,
                       linear_response = linear_response, dsg_matrix = design_matrix_final, y = y_data)

        sess$close()

        class(result) <- "MLEtf"

        return(result)
}


#------------------------------------------------------------------------
# Function to create design matrix --------------------------------------
#------------------------------------------------------------------------
model_matrix_GLM <- function(formula, data){

        # Extract the right side of formulas
        formulas_corrector <- all.vars(formula)[-1]
        formulas_tmp <- as.list(formulas_corrector)
        #print(formulas_tmp)

        # check that all explanatory variables are included in data
        dep_variables <- vector(mode = "list", length = length(formulas_tmp))
        dep_variables <- lapply(1:length(formulas_corrector), FUN = function(x) dep_variables[[x]] = trimws(formulas_corrector[x]))
        dep_variables <- unlist(dep_variables)
        #print(dep_variables)
        dep_variables <- dep_variables[!dep_variables == 1]
        #print(dep_variables)

        if ((ncol(data) - 1) < length(dep_variables)) {
                stop(paste0("Data for some explanatory variables is missing. Please include it\n",
                            "in the dataframe provided in the 'data' argument."))
        } else if ((ncol(data) - 1) > length(dep_variables)) {
                stop(paste0("There are more variables that expected in the dataframe provided in the 'data' argument. \n",
                            "Please check your input data or the formula provided."))
        }

        # Build design matrix
        data_reg <- model.frame(formula, data)
        response <- data_reg[, 1]
        dsg_matrix <- model.matrix(object = formula, data = data_reg)
        return(list(design_matrix = dsg_matrix, y = response, data_reg = data_reg))
}



#------------------------------------------------------------------------
# TensorFlow family -----------------------------------------------------
#------------------------------------------------------------------------
tf_family <- function(family, link_function) {

        if (is.null(link_function) & (family == "Normal" | family == "Poisson")) {
                list_link <- list(Normal = "identity",
                                  Poisson = "log")

                link_function <- list_link[[family]]
        }

        sel_families <- ifelse(family == "Bernoulli" & link_function == "sigmoid", "Bernoulli",
                               ifelse(family == "Bernoulli" & link_function == "normalCDF", "BernoulliNormalCDF",
                                      ifelse(family == "Gamma" & link_function == "log", "GammaExp",
                                             ifelse(family == "Gamma" & link_function == "softplus", "GammaSoftplus",
                                                    ifelse(family == "LogNormal" & link_function == "log", "LogNormal",
                                                           ifelse(family == "LogNormal" & link_function == "softplus", "LogNormalSoftplus",
                                                                  ifelse(family == "Normal" & link_function == "identity", "Normal",
                                                                         ifelse(family == "Normal" & link_function == "inverse", "NormalReciprocal",
                                                                                ifelse(family == "Poisson" & link_function == "log", "Poisson",
                                                                                       ifelse(family == "Poisson" & link_function == "softplus", "PoissonSoftplus", "NotAvailable"))))))))))


        return(sel_families)

}

