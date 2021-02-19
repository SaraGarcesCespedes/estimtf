#' @title summary.MLEtf function
#'
#' @description Function to produce result summaries of the estimates of parameters from statistical
#' distributions using \code{\link{dist_estimtf2}} or parameters from regression models using
#' \code{\link{reg_estimtf}}.
#'
#' @author Sara Garces Cespedes
#'
#' @param object an object of class \code{MLEtf} for which a summary is desired.
#' @param ... additional arguments affecting the summary produced.
#'
#' @return The output from
#'
#' @details \code{summary.MLEtf} function displays estimates and standard errors of parameters from statistical
#' distributions and regression models. Also, for regression parameters, this function computes and displays
#' the z values and p-values of significance tests for these parameters.
#'
#' @importFrom stats printCoefmat
#' @importFrom stats pnorm
#' @importFrom stats pt
#'
#' @examples
#' #-------------------------------------------------------------
#' # Estimation of both normal distrubution parameters
#' x <- rnorm(n = 1000, mean = 10, sd = 3)
#'
#' summary(dist_estimtf2(x, xdist = "Normal", optimizer = "AdamOptimizer",
#'                            hyperparameters = list(learning_rate = 0.1)))
#'
#' @rdname summary.MLEtf
#' @export
#------------------------------------------------------------------------
# Summary function ------------------------------------------------------
#------------------------------------------------------------------------
summary.MLEtf <- function(object, ...) {

        estimates <- as.numeric(object$outputs$estimates)
        if (object$outputs$type != "MLEdistf_fdp") {
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
        #pvalue <- as.numeric(2 * pnorm(abs(zvalue), lower.tail = FALSE))
        n <- object$outputs$n
        p <-length(estimates)
        df <- n - (p)
        pvalue <- as.numeric(2 * pt(abs(zvalue), df, lower.tail = FALSE))

        if (object$outputs$type == "MLEdistf" | object$outputs$type == "MLEdistf_fdp") {
                if (object$outputs$type == "MLEdistf") {
                        cat(paste0('Distribution: ', object$distribution),'\n')
                }
                cat(paste0('Number of observations: ', object$outputs$n),'\n')
                cat(paste0('TensorFlow optimizer: ', object$optimizer),'\n')
                cat("---------------------------------------------------\n")
                restable <- cbind(estimate = estimates, stderror = stderror)
                restable <- data.frame(restable)
                colnames(restable) <- c('Estimate ', 'Std. Error')
                rownames(restable) <- object$outputs$parnames
                printCoefmat(restable, digits = 4)
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
                colnames(restable) <- c('Estimate ', 'Std. Error', 't value', 'Pr(>|t|)')

                resparam <- restable[(1 + t[[1]]):(t[[1]] + object$outputs$nbetas[[1]]), ]
                resparam <- data.frame(resparam)
                rownames(resparam) <- object$output$names[(1 + t[[1]]):(t[[1]] + object$outputs$nbetas[[1]])]
                printCoefmat(resparam, digits = 4, P.values = TRUE, has.Pvalue = TRUE)
                cat("----------------------------------------------------------------\n")

        } else {
                t <- vector(mode = "list")
                print(object$outputs$np)
                if (object$outputs$np > 1) {
                        t <- lapply(1:object$outputs$np,
                                    FUN = function(i) t[[i]] <- ifelse(i == 1, 0,
                                                                       Reduce("+",
                                                                              object$outputs$nbetas[[1:(i - 1)]])))
                } else {
                        t[[1]] <- 0
                }
                cat(paste0('Distribution of response variable: ', object$distribution),'\n')
                cat(paste0('Number of observations: ', object$outputs$n),'\n')
                cat(paste0('TensorFlow optimizer: ', object$optimizer),'\n')
                cat("----------------------------------------------------------------\n")
                restable <- cbind(estimate = estimates, stderror = stderror, zvalue = zvalue,
                                  pvalue = pvalue)
                restable <- data.frame(restable)
                colnames(restable) <- c('Estimate ', 'Std. Error', 't value', 'Pr(>|t|)')
                for (i in 1:object$outputs$np) {
                        cat(paste0('Distributional parameter: ',
                                   names(object$dsgmatrix)[i],'\n'))
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
#' @description Function to display the estimates of parameters from statistical
#' distributions using \code{\link{dist_estimtf2}} or parameters from regression models using
#' \code{\link{reg_estimtf}}.
#'
#' @author Sara Garces Cespedes
#'
#' @param x an object of class \code{MLEtf} for which a summary is desired.
#' @param ... additional arguments affecting the summary produced.
#'
#'
#' @details \code{print.MLEtf} function displays estimates of parameters from statistical distributions
#' and regression models.
#'
#' @importFrom stats printCoefmat
#'
#' @examples
#' #-------------------------------------------------------------
#' # Estimation of both normal distrubution parameters
#' x <- rnorm(n = 1000, mean = 10, sd = 3)
#'
#' print(dist_estimtf2(x, xdist = "Normal", optimizer = "AdamOptimizer",
#'                            hyperparameters = list(learning_rate = 0.1)))
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

        } else if (object$outputs$type == "MLEregtf") {
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
                for (i in 1:object$outputs$np) {
                        cat(paste0('Distributional parameter: ',
                                   names(object$dsgmatrix)[i],'\n'))
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
#' @description Function to display a graph that contains the loss value in each iteration of
#' the estimation process using \code{\link{dist_estimtf2}} function or using \code{\link{reg_estimtf}} function.
#'
#' @author Sara Garces Cespedes
#'
#' @param object an object of class \code{MLEtf} for which a plot with loss values is desired.
#' @param ... additional arguments affecting the summary produced.
#'
#'
#' @details \code{plot_loss.MLEtf} function displays a graph with the loss values in each iteration of the
#' estimation process for distributional or regression parameters.
#'
#' @import ggplot2
#'
#' @examples
#' #-------------------------------------------------------------
#' # Estimation of both normal distrubution parameters
#' x <- rnorm(n = 1000, mean = 10, sd = 3)
#'
#' plot_loss(dist_estimtf2(x, xdist = "Normal", optimizer = "AdamOptimizer",
#'                            hyperparameters = list(learning_rate = 0.1)))
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

        ggplot2::ggplot(data = loss_values, aes(x = Iteration, y = loss)) +
                geom_line() +
                geom_point() +
                theme(plot.title = element_text(face = "bold", size =16, hjust = 0.5)) +
                labs(y = "Loss value", x = "Iteration", title = "Loss value in each iteration of the estimation process")



}

