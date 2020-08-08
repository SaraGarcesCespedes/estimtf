summary.MLEtf <- function(object, ...) {

        estimates <- as.numeric(object$outputs$estimates)
        stderror <- unlist(object$stderrt, use.names = FALSE)
        zvalue <- as.numeric(estimates / stderror)
        pvalue <- as.numeric(2 * pnorm(abs(zvalue), lower.tail = FALSE))

        t <- vector(mode = "list")
        if (object$outputs$np > 1) {
                t <- lapply(1:object$outputs$np,
                            FUN = function(i) t[[i]] <- ifelse(i == 1, 0,
                                                               Reduce("+",
                                                                      object$outputs$nbetas[[1:(i - 1)]])))
        } else {
                t[[1]] <- 0
        }

        if (object$outputs$type == "MLEdistf") {
                cat(paste0('Distribution: ', object$distribution),'\n')
                cat(paste0('Number of observations: ', object$outputs$n),'\n')
                cat(paste0('TensorFlow optimizer: ', object$optimizer),'\n')
                cat("---------------------------------------------------\n")
                restable <- cbind(estimate = estimates, stderror = stderror)
                restable <- data.frame(restable)
                colnames(restable) <- c('Estimate ', 'Std. Error')
                printCoefmat(restable, digits = 4)

        } else {
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
summary(a)


