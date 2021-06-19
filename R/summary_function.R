#' @title summary.MLEtf function
#'
#' @description Function to produce result summaries of the estimates of parameters from probability
#' distributions using the \code{\link{mle_tf}} function or parameters from regression models using
#' the \code{\link{mlereg_tf}} function.
#'
#' @author Sara Garcés Céspedes \email{sgarcesc@unal.edu.co}
#'
#' @param object an object of class \code{MLEtf} for which a summary is desired.
#' @param ... additional arguments affecting the summary produced.
#'
#'
#' @details \code{summary.MLEtf} function displays estimates and standard errors of parameters from statistical
#' distributions and regression models. Also, this function computes and displays the Z-score and p-values of significance
#' tests for these parameters.
#'
#' @importFrom stats printCoefmat
#' @importFrom stats pnorm
#' @importFrom stats pt
#'
#' @examples
#' #---------------------------------------------------------------
#' # Estimation of both normal distrubution parameters
#'
#' # Generate a sample from the normal distribution
#' x <- rnorm(n = 1000, mean = 10, sd = 3)
#'
#' # Use the summary function
#' summary(mle_tf(x, xdist = "Normal",
#'                optimizer = "AdamOptimizer",
#'                initparam = list(mean = 1.0, sd = 1.0),
#'                hyperparameters = list(learning_rate = 0.1)))
#'
#' @rdname summary.MLEtf
#' @export
#------------------------------------------------------------------------
# Summary function ------------------------------------------------------
#------------------------------------------------------------------------
summary.MLEtf <- function(object, ...) {

        estimates <- as.numeric(object$outputs$estimates)
        if (object$outputs$type != "MLEdistf_fdp" & object$outputs$type != "MLEreg_fdp") {
                dist <- object$distribution
        }

        if (object$outputs$type == "MLEglmtf") {
                if (dist == "Binomial") {
                        dsg_matrix <- object$dsgmatrix
                        X <- dsg_matrix$logits
                        fitted_values <- X %*% estimates
                        fitted_values <- exp(fitted_values) / (1 + exp(fitted_values))
                        diagonal <- c(fitted_values * (1 - fitted_values))
                        V <- diag(diagonal)
                        cov_matrix <- solve(t(X) %*% V %*% X)
                        stderror <- diag(sqrt(cov_matrix))
                } else {
                        stderror <- unlist(object$stderrt, use.names = FALSE)
                }
        } else {
                stderror <- unlist(object$stderrt, use.names = FALSE)
        }

        zvalue <- as.numeric(estimates / stderror)
        pvalue <- as.numeric(2 * pnorm(abs(zvalue), lower.tail = FALSE))
        n <- object$outputs$n
        p <-length(estimates)
        df <- n - (p)
        #pvalue <- as.numeric(2 * pt(abs(zvalue), df, lower.tail = FALSE))

        if (object$outputs$type == "MLEdistf" | object$outputs$type == "MLEdistf_fdp") {
                if (object$outputs$type == "MLEdistf") {
                        cat(paste0('Distribution: ', object$distribution),'\n')
                }
                cat(paste0('Number of observations: ', object$outputs$n),'\n')
                cat(paste0('TensorFlow optimizer: ', object$optimizer),'\n')
                cat("---------------------------------------------------\n")
                restable <- cbind(estimate = estimates, stderror = stderror, zvalue = zvalue,
                                  pvalue = pvalue)
                restable <- data.frame(restable)
                colnames(restable) <- c('Estimate ', 'Std. Error', 'Z value', 'Pr(>|z|)')
                rownames(restable) <- object$outputs$parnames
                printCoefmat(restable, digits = 4, P.values = TRUE, has.Pvalue = TRUE)
        } else if (object$outputs$type == "MLEglmtf") {
                t <- vector(mode = "list")
                t[[1]] <- 0
                cat(paste0('Family: ', object$distribution),'\n')
                cat(paste0('Number of observations: ', object$outputs$n),'\n')
                cat(paste0('TensorFlow optimizer: ', object$optimizer),'\n')
                cat("----------------------------------------------------------------\n")
                restable <- cbind(estimate = estimates, stderror = stderror, zvalue = zvalue,
                                  pvalue = pvalue)
                restable <- data.frame(restable)
                colnames(restable) <- c('Estimate ', 'Std. Error', 'Z value', 'Pr(>|z|)')

                resparam <- restable[(1 + t[[1]]):(t[[1]] + object$outputs$nbetas[[1]]), ]
                resparam <- data.frame(resparam)
                rownames(resparam) <- object$output$names[(1 + t[[1]]):(t[[1]] + object$outputs$nbetas[[1]])]
                printCoefmat(resparam, digits = 4, P.values = TRUE, has.Pvalue = TRUE)
                cat("----------------------------------------------------------------\n")

        } else {
                t <- vector(mode = "list")
                if (object$outputs$np > 1) {
                        t <- lapply(1:object$outputs$np,
                                    FUN = function(i) t[[i]] <- ifelse(i == 1, 0,
                                                                       Reduce("+",
                                                                              object$outputs$nbetas[[1:(i - 1)]])))
                } else {
                        t[[1]] <- 0
                }
                if (object$outputs$type == "MLEregtf") {
                        cat(paste0('Distribution: ', object$distribution),'\n')
                }
                cat(paste0('Number of observations: ', object$outputs$n),'\n')
                cat(paste0('TensorFlow optimizer: ', object$optimizer),'\n')
                cat("----------------------------------------------------------------\n")
                restable <- cbind(estimate = estimates, stderror = stderror, zvalue = zvalue,
                                  pvalue = pvalue)
                restable <- data.frame(restable)
                colnames(restable) <- c('Estimate ', 'Std. Error', 'Z value', 'Pr(>|z|)')
                names_param <- object$outputs$names_regparam
                for (i in 1:object$outputs$np) {
                        cat(paste0('Distributional parameter: ',
                                   names_param[i],'\n'))
                        cat("----------------------------------------------------------------\n")
                        resparam <- restable[(1 + t[[i]]):(t[[i]] + object$outputs$nbetas[[i]]), ]
                        resparam <- data.frame(resparam)
                        rownames(resparam) <- object$output$names[(1 + t[[i]]):(t[[i]] + object$outputs$nbetas[[i]])]
                        printCoefmat(resparam, digits = 4, P.values = TRUE, has.Pvalue = TRUE)
                        cat("----------------------------------------------------------------\n")
                }


        }

}

#' @title print.MLEtf function
#'
#' @description Function to display the estimates of parameters from probability
#' distributions using the \code{\link{mle_tf}} function or parameters from regression models using
#' the \code{\link{mlereg_tf}} function.
#'
#' @author Sara Garcés Céspedes \email{sgarcesc@unal.edu.co}
#'
#' @param x an object of class \code{MLEtf} for which a summary is desired.
#' @param ... additional arguments affecting the summary produced.
#'
#'
#' @details \code{print.MLEtf} function displays the estimates of parameters from probability distributions
#' and regression models.
#'
#' @importFrom stats printCoefmat
#'
#' @examples
#' #---------------------------------------------------------------
#' # Estimation of both normal distrubution parameters
#'
#' # Generate a sample from the normal distribution
#' x <- rnorm(n = 1000, mean = 10, sd = 3)
#'
#' # Use the print function
#' print(mle_tf(x, xdist = "Normal",
#'              initparam = list(mean = 1.0, sd = 1.0),
#'               optimizer = "AdamOptimizer",
#'               hyperparameters = list(learning_rate = 0.1)))
#'
#' @rdname print.MLEtf
#' @export
#------------------------------------------------------------------------
# Print function --------------------------------------------------------
#------------------------------------------------------------------------

print.MLEtf <- function(x, ...) {

        object <- x
        estimates <- as.numeric(object$outputs$estimates)


        if (object$outputs$type == "MLEdistf" | object$outputs$type == "MLEdistf_fdp") {
                cat(paste0('Estimates:','\n'))
                estimates <- as.data.frame(estimates)
                restable <- data.frame(t(estimates))
                colnames(restable) <- object$outputs$parnames
                rownames(restable) <- ""
                printCoefmat(restable, digits = 4)
                cat("---------------------------------------------------\n")
                cat(paste0(object$outputs$convergence),'\n')

        } else if (object$outputs$type == "MLEregtf" | object$outputs$type == "MLEregtf_fdp") {
                t <- vector(mode = "list")
                if (object$outputs$np > 1) {
                        t <- lapply(1:object$outputs$np,
                                    FUN = function(i) t[[i]] <- ifelse(i == 1, 0,
                                                                       Reduce("+",
                                                                              object$outputs$nbetas[[1:(i - 1)]])))
                } else {
                        t[[1]] <- 0
                }
                cat(paste0(object$outputs$convergence),'\n')
                cat("---------------------------------------------------\n")
                restable <- data.frame(t(estimates))
                names_param <- object$outputs$names_regparam
                for (i in 1:object$outputs$np) {
                        cat(paste0('Distributional parameter: ',
                                   names_param[i],'\n'))
                        resparam <- restable[, (1 + t[[i]]):(t[[i]] + object$outputs$nbetas[[i]])]
                        resparam <- as.data.frame(resparam)
                        colnames(resparam) <- object$output$names[(1 + t[[i]]):(t[[i]] + object$outputs$nbetas[[i]])]
                        rownames(resparam) <- ""
                        printCoefmat(resparam, digits = 4)
                        cat("---------------------------------------------------\n")
                }


        } else if (object$outputs$type == "MLEglmtf"){
                t <- vector(mode = "list")
                t[[1]] <- 0

                cat(paste0(object$outputs$convergence),'\n')
                cat("---------------------------------------------------\n")
                restable <- data.frame(t(estimates))
                resparam <- restable[, (1 + t[[1]]):(t[[1]] + object$outputs$nbetas[[1]])]
                resparam <- as.data.frame(resparam)
                colnames(resparam) <- object$output$names[(1 + t[[1]]):(t[[1]] + object$outputs$nbetas[[1]])]
                rownames(resparam) <- ""
                printCoefmat(resparam, digits = 4)
                cat("---------------------------------------------------\n")



        } else if (object$outputs$type == "MLEglm") {
                cat(paste0('Estimates:','\n'))
                estimates <- object$coeff_estim
                restable <- data.frame(t(estimates))
                colnames(restable) <- object$names_coeff
                rownames(restable) <- ""
                printCoefmat(restable, digits = 4)
                cat("---------------------------------------------------\n")
                cat(paste0("Converged: ", object$converged),'\n')

        } else if (object$outputs$type == "estim") {
                cat(paste0('Estimates:','\n'))
                estimates <- as.data.frame(estimates)
                restable <- data.frame(t(estimates))
                colnames(restable) <- object$outputs$parnames
                rownames(restable) <- ""
                printCoefmat(restable, digits = 4)
                cat("---------------------------------------------------\n")
                cat(paste0(object$outputs$convergence),'\n')
        }



}


#' @title plot_loss function
#'
#' @description Function to display a graph that contains the loss value computed in each iteration of
#' the optimization process performed using the \code{\link{mle_tf}} function or using the \code{\link{mlereg_tf}}
#' function.
#'
#' @author Sara Garcés Céspedes \email{sgarcesc@unal.edu.co}
#'
#' @param object an object of class \code{MLEtf} for which a plot with loss values is desired.
#' @param ... additional arguments affecting the summary produced.
#'
#'
#' @details \code{plot_loss.MLEtf} function displays a graph with the loss values, which correspond to the
#' negative log-likelihood computed in each iteration of the optimization process.
#'
#'
#' @examples
#' #---------------------------------------------------------------
#' # Estimation of both normal distrubution parameters
#'
#' # Generate a sample from the normal distribution
#' x <- rnorm(n = 1000, mean = 10, sd = 3)
#'
#' # Use the plot_loss function
#' plot_loss(mle_tf(x, xdist = "Normal",
#'                  optimizer = "AdamOptimizer",
#'                  initparam = list(mean = 1.0, sd = 1.0),
#'                  hyperparameters = list(learning_rate = 0.1)))
#'
#' @export
#------------------------------------------------------------------------
# Loss function ---------------------------------------------------------
#------------------------------------------------------------------------

plot_loss <- function(object, ...) {

        loss_values <- as.data.frame(object$tf)
        loss_values["Iteration"] <- 1:nrow(loss_values)
        loss_values <- loss_values %>% select(loss, Iteration)
        loss_values$loss <- as.numeric(loss_values$loss)

        plot(loss_values$loss, type = "o", col = "red", xlab = "Iteration", ylab = "Loss",
             main = "Loss value obtained in each iteration")


}

