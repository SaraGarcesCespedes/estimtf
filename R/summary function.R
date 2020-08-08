#' @title summary.MLEtf function
#'
#' @description Function to produce result summaries of the estimates of distributional
#' parameters using \code{\link{dist_estimtf}} or regression parameters using \code{\link{reg_estimtf}}.
#'
#' @author Sara Garces Cespedes
#'
#' @param object an object of class \code{MLEtf} for which a summary is desired.
#' @param ... additional arguments affecting the summary produced.
#'
#' @return The output from
#'
#' @details \code{summary.MLEtf} function displays estimates and standard errors of distributional and
#' regression parameters. Also, for regression parameters, this function computes and displays the z-score
#' and p-values of significance tests for these parameters.
#'
#' @examples
#' #-------------------------------------------------------------
#' # Estimation of both normal distrubution parameters
#' x <- rnorm(n = 1000, mean = 10, sigma = 3)
#'
#' summary(dist_estimtf(x, xdist = "Normal", optimizer = "AdamOptimizer",
#'                            hyperparameters = list(learning_rate = 0.1)))
#'
#' @export
#------------------------------------------------------------------------
# Summary function ------------------------------------------------------
#------------------------------------------------------------------------
summary.MLEtf <- function(object, ...) {

        estimates <- as.numeric(object$outputs$estimates)
        stderror <- unlist(object$stderrt, use.names = FALSE)
        zvalue <- as.numeric(estimates / stderror)
        pvalue <- as.numeric(2 * pnorm(abs(zvalue), lower.tail = FALSE))



        if (object$outputs$type == "MLEdistf") {
                cat(paste0('Distribution: ', object$distribution),'\n')
                cat(paste0('Number of observations: ', object$outputs$n),'\n')
                cat(paste0('TensorFlow optimizer: ', object$optimizer),'\n')
                cat("---------------------------------------------------\n")
                restable <- cbind(estimate = estimates, stderror = stderror)
                restable <- data.frame(restable)
                colnames(restable) <- c('Estimate ', 'Std. Error')
                rownames(restable) <- object$outputs$parnames
                printCoefmat(restable, digits = 4)

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
                cat(paste0('Distribution of response variable: ', object$distribution),'\n')
                cat(paste0('Number of observations: ', object$outputs$n),'\n')
                cat(paste0('TensorFlow optimizer: ', object$optimizer),'\n')
                cat("----------------------------------------------------------------\n")
                restable <- cbind(estimate = estimates, stderror = stderror, zvalue = zvalue,
                                  pvalue = pvalue)
                restable <- data.frame(restable)
                colnames(restable) <- c('Estimate ', 'Std. Error', 'z value', 'Pr(>|z|)')
                for (i in 1:object$outputs$np) {
                        cat(paste0('Distributional parameter: ',
                                   names(object$dsgmatrix)[i],'\n'))
                        cat("----------------------------------------------------------------\n")
                        resparam <- restable[(1 + t[[i]]):(t[[i]] + object$outputs$nbetas[[i]]), ]
                        rownames(resparam) <- object$output$names[(1 + t[[i]]):(t[[i]] + object$outputs$nbetas[[i]])]
                        printCoefmat(resparam, digits = 4, P.values = TRUE, has.Pvalue = TRUE)
                        cat("----------------------------------------------------------------\n")
                }


        }

}

#------------------------------------------------------------------------
# Print function --------------------------------------------------------
#------------------------------------------------------------------------

print.MLEtf <- function(object, ...) {

        estimates <- as.numeric(object$outputs$estimates)


        if (object$outputs$type == "MLEdistf") {
                cat(paste0('Estimates:','\n'))
                restable <- data.frame(t(estimates))
                colnames(restable) <- object$outputs$parnames
                rownames(restable) <- ""
                printCoefmat(restable, digits = 4)
                cat("---------------------------------------------------\n")
                cat(paste0(object$outputs$convergence),'\n')

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
                cat(paste0(object$outputs$convergence),'\n')
                cat("---------------------------------------------------\n")
                restable <- data.frame(t(estimates))
                for (i in 1:object$outputs$np) {
                        cat(paste0('Distributional parameter: ',
                                   names(object$dsgmatrix)[i],'\n'))
                        resparam <- restable[, (1 + t[[i]]):(t[[i]] + object$outputs$nbetas[[i]])]
                        colnames(resparam) <- object$output$names[(1 + t[[i]]):(t[[i]] + object$outputs$nbetas[[i]])]
                        rownames(resparam) <- ""
                        printCoefmat(resparam, digits = 4)
                        cat("---------------------------------------------------\n")
                }


        }



}


#------------------------------------------------------------------------
# Plot function ---------------------------------------------------------
#------------------------------------------------------------------------

