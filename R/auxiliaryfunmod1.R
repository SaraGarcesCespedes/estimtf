#------------------------------------------------------------------------
# Estimation of distribution parameters (disable eager execution) -------
#------------------------------------------------------------------------

disableagerdist <- function(x, dist, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np, distnotf, xdist, optimizer, limits) {

        # Disable eager execution
        tensorflow::tf$compat$v1$disable_eager_execution()

        # Create placeholder
        X <- tensorflow::tf$compat$v1$placeholder(dtype=tf$float32, name = "x_data")

        # Create list to store the parameters to be estimated
        var_list <- vector(mode = "list", length = np)
        link <- vector(mode = "list", length = np)
        var_list_new <- vector(mode = "list", length = np)

        # Create tf Variables
        var_list <- lapply(1:np,
                           FUN = function(i) var_list[[i]] <- assign(names(initparam)[i],
                                                                     tensorflow::tf$Variable(initparam[[i]],
                                                                                             dtype = tf$float32,
                                                                                             name = names(initparam)[i]),
                                                                     envir = .GlobalEnv))

        names(var_list) <- names(initparam)


        if (!is.null(limits)) {
                link <- lapply(1:np, FUN = function(i) link[[i]] <- link_dist(limits[[i]], var_list[[i]], names(var_list)[i]))
                var_list_new <- lapply(1:np, FUN = function(i) var_list_new[[i]] <- assign(names(initparam)[i], link[[i]], envir = .GlobalEnv))
                names(var_list_new) <- names(initparam)
        } else {
                var_list_new <- var_list
        }




        # Create a list with all parameters, fixed and not fixed
        vartotal <- append(fixparam, var_list_new)

        # Create vectors to store parameters, gradientes and loss values of each iteration
        loss <- new_list <- parameters <- gradients <- itergrads <- objvariables <- vector(mode = "list")

        # Create list with variables without names
        new_list <- lapply(1:np, FUN = function(i) new_list[[i]] <- var_list[[i]])

        n <- length(x)

        # Define loss function depending on the distribution
        if (xdist %in% distnotf) {
                loss_value <- lossfun_mle(dist, vartotal, X, n)
        } else {
                density <- do.call(what = dist, vartotal)
                loss_value <- tensorflow::tf$negative(tensorflow::tf$reduce_sum(density$log_prob(value = X)))
        }


        # Compute gradients
        grads <- tensorflow::tf$gradients(loss_value, new_list)

        # Define optimizer
        if (optimizer == "GradientDescentOptimizer") {
                global_step <- tensorflow::tf$Variable(0, trainable = FALSE)
                starter_learning_rate <- hyperparameters$learning_rate
                learning.rate <- tensorflow::tf$compat$v1$train$exponential_decay(starter_learning_rate, global_step,
                                                                                  100000, 0.96, staircase=TRUE)
                hyperparameters$learning_rate <- learning.rate
                seloptimizer <- do.call(what = opt, hyperparameters)
                train <- eval(parse(text = "seloptimizer$minimize(loss_value, global_step = global_step)"))

        } else {
                seloptimizer <- do.call(what = opt, hyperparameters)
                train <- eval(parse(text = "seloptimizer$minimize(loss_value)"))
        }

        # Initialize the variables and open the session
        init <- tensorflow::tf$compat$v1$initialize_all_variables()
        sess <- tensorflow::tf$compat$v1$Session()
        sess$run(init)

        # Create dictionary to feed data into graph
        fd <- dict(X = x)

        # Initialize step
        step <- 0

        while(TRUE){
                # Update step
                step <- step + 1

                # Gradient step
                sess$run(train, feed_dict = fd)

                objvariables <- lapply(1:np, FUN = function(i) objvariables[[i]] <- as.numeric(sess$run(new_list[[i]])))
                itergrads <- lapply(1:np, FUN = function(i) itergrads[[i]] <- as.numeric(sess$run(grads, feed_dict = fd)[[i]]))

                parameters[[step]] <- objvariables
                gradients[[step]] <- itergrads

                # Save loss value
                loss[[step]] <- as.numeric(sess$run(loss_value, feed_dict = fd))

                if (is.na(loss[[step]])){
                        stop(paste0("The process failed because the loss value in the last iterarion is NaN \n",
                                    "Follow these recommendations and start the process again: \n",
                                    "1. Reduce the learning rate. \n",
                                    "2. Check your input data as it is possible that some of the values are neither \n",
                                    "integer nor float. \n",
                                    "3. Try different optimizers. \n",
                                    "4. Change the initial values provided for the parameters. \n",
                                    "5. Scale your data differently as this problem may happen because your input values \n",
                                    "are too high."))
                }

                # Conditions
                if (step != 1) {
                        if (abs(loss[[step]] - loss[[step-1]]) < tolerance$loss){
                                convergence <- paste("Loss function convergence,", step, "iterations needed.")
                                break
                        } else if (step >= maxiter) {
                                convergence <- paste("Maximum number of iterations reached.")
                                break
                        } else if (isTRUE(sapply(1:np, FUN= function(x) abs(parameters[[step]][[x]]-parameters[[step-1]][[x]]) < tolerance$parameters))) {
                                convergence <- paste("Parameters convergence,", step, "iterations needed.")
                                break
                        } else if (isTRUE(sapply(1:np, FUN= function(x) abs(gradients[[step]][[x]]-gradients[[step-1]][[x]]) < tolerance$gradients))) {
                                convergence <- paste("Gradients convergence,", step, "iterations needed.")
                                break
                        }
                }
        }

        # Compute Hessian matrix
        hesslist <- stderror <- vector(mode = "list", length = np)
        hesslist <- lapply(1:np, FUN = function(i) hesslist[[i]] <- tensorflow::tf$gradients(grads[[i]], new_list))
        hess <- tensorflow::tf$stack(values=hesslist, axis=0)
        mhess <- sess$run(hess, feed_dict = fd)
        diagvarcov <- hessian_matrix_try(mhess)
        #diagvarcov <- sqrt(diag(solve(mhess)))

        if (!is.null(diagvarcov)) {
                stderror <- lapply(1:np, FUN = function(i) stderror[[i]] <- diagvarcov[i])
                names(stderror) <- names(var_list)
        } else {
                stderror <- lapply(1:np, FUN = function(i) stderror[[i]] <- NULL)
                names(stderror) <- names(var_list)
        }


        # Close tf session
        sess$close()

        # Organize results of each iteration
        gradients <- purrr::transpose(gradients)
        parameters <- purrr::transpose(parameters)
        gradientsfinal <- parametersfinal <- namesgradients <- as.numeric()
        gradientsfinal <- sapply(1:np, FUN = function(i) gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[i]])))
        parametersfinal <- sapply(1:np, FUN = function(i) parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[i]])))
        namesgradients <- sapply(1:np, FUN = function(i) namesgradients <- cbind(namesgradients, paste0("Gradients ", names(var_list)[i])))

        # change name of parameters to match R parameters
        if (!xdist %in% distnotf) {
                names_param <- names(initparam)
                names_new <- vector(mode = "numeric", length = length(names_param))
                names_new <- sapply(1:length(names_param), FUN = function(i) names_new[i] <- parameter_name_R(names_param[i], xdist))
                names(initparam) <- names_new
        }
        # Table of results
        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(var_list), namesgradients)

        # transform estimates
        estimates <- tail(results.table[, 2:(np + 1)], 1)

        if (!is.null(limits)) {
                estimates_final <- as.numeric()
                stderror_final <- vector(mode = "list", length = np)
                results_transform <- vector(mode = "list", length = np)
                results_transform <- lapply(1:np, FUN = function(i) results_transform[[i]] <- transform(limits[[i]], estimates[i], stderror[[i]]))
                estimates_final <- sapply(1:np, FUN = function(i) estimates_final[i] <- results_transform[[i]][["estimate_final"]])
                stderror_final <- lapply(1:np, FUN = function(i) stderror_final[[i]] <- results_transform[[i]][["stderror_final"]])
        } else {
                estimates_final <- estimates
                stderror_final <- stderror
        }


        outputs <- list(n = n, type = "MLEdistf", parnames = names(initparam),
                        estimates = estimates_final,
                        convergence = convergence)
        result <- list(results = results.table, vcov = mhess, standarderror = stderror_final,
                       outputs = outputs)
        return(result)
}
#------------------------------------------------------------------------
# Link function ---------------------------------------------------------
#------------------------------------------------------------------------
# Methodology taken from Nonlinear Parameter Optimization Using R Tools
transform <- function(limits, estimate, stderror) {


        qinv_transform <- function(x) {
                qix <- x
                if (c1 == TRUE) {
                        qix <- lower + 0.5 * (upper - lower) * (1 + tanh(x))
                        stderror_final <- NULL
                } else if (c2 == TRUE) {
                        qix <- x
                        stderror_final <- stderror
                } else if (c3 == TRUE) {
                        qix <- lower + exp(x)
                        stderror_final <- NULL
                } else if (c4 == TRUE) {
                        qix <- upper - exp(x)
                        stderror_final <- NULL
                }
                return(list(qix = qix, stderror_final = stderror_final))
        }

        if (length(limits) == 1) {
                estimate_final <- estimate
                stderror_final <- stderror
        } else {
                if (any(is.na(limits))) {
                        stop("Any NAs not allowed in bounds.")
                } else if (any(is.null(limits))) {
                        stop("Any NULLs not allowed in bounds.")
                } else if (limits[1] == limits[2]) {
                        stop("No component of bounds must be equal.")
                } else if (limits[1] > limits[2]) {
                        stop("Lower bound can not be greater than upper bound.")
                } else {
                        lower <- limits[1]
                        upper <- limits[2]

                        low.finite <- is.finite(lower)
                        upp.finite <- is.finite(upper)
                        c1 <- low.finite & upp.finite # both lower and upper bounds are finite
                        c2 <- !(low.finite | upp.finite) # both lower and upper bounds infinite
                        c3 <- !(c1 | c2) & low.finite # finite lower bound, infinite upper bound
                        c4 <- !(c1 | c2) & upp.finite # finite upper bound, infinite lower bound

                        results <- qinv_transform(estimate)
                        estimate_final <- results[["qix"]]
                        stderror_final <- results[["stderror_final"]]
                }
        }
        return(list(estimate_final = estimate_final, stderror_final = stderror_final))
}

link_dist <- function(limits, param_tf, param_name) {


        qinv_link_dist <- function(x) {
                qix <- x
                if (c1 == TRUE) {
                        qix <- lower + 0.5 * (upper - lower) * (1 + tf$math$tanh(x))
                } else if (c2 == TRUE) {
                        qix <- x
                } else if (c3 == TRUE) {
                        qix <- lower + tf$math$exp(x)
                } else if (c4 == TRUE) {
                        qix <- upper - tf$math$exp(x)
                }
                return(qix)
        }

        if (length(limits) == 1) {
                param_final <- param_tf
        } else {
                if (any(is.na(limits))) {
                        stop("Any NAs not allowed in bounds.")
                } else if (any(is.null(limits))) {
                        stop("Any NULLs not allowed in bounds.")
                } else if (limits[1] == limits[2]) {
                        stop("No component of bounds must be equal.")
                } else if (limits[1] > limits[2]) {
                        stop("Lower bound can not be greater than upper bound.")
                } else {
                        lower <- limits[1]
                        upper <- limits[2]

                        low.finite <- is.finite(lower)
                        upp.finite <- is.finite(upper)
                        c1 <- low.finite & upp.finite # both lower and upper bounds are finite
                        c2 <- !(low.finite | upp.finite) # both lower and upper bounds infinite
                        c3 <- !(c1 | c2) & low.finite # finite lower bound, infinite upper bound
                        c4 <- !(c1 | c2) & upp.finite # finite upper bound, infinite lower bound

                        param_final <- qinv_link_dist(param_tf)
                }
        }
        return(param_final)
}

#------------------------------------------------------------------------
# Loss function for distributions not included in TF --------------------
#------------------------------------------------------------------------
lossfun_mle <- function(dist, vartotal, X, n) {

        if (dist == "FWE") {
                loss <- -tensorflow::tf$reduce_sum(tensorflow::tf$math$log(vartotal[["mu"]] + vartotal[["sigma"]] / (X ^ 2))) -
                        tensorflow::tf$reduce_sum(vartotal[["mu"]] * X - vartotal[["sigma"]] / X) +
                        tensorflow::tf$reduce_sum(tensorflow::tf$math$exp(vartotal[["mu"]] * X - vartotal[["sigma"]] / X))
        } else if (dist == "InstantaneousFailures") {
                loss <- -tensorflow::tf$reduce_sum(tensorflow::tf$math$log((((vartotal[["lambda"]] ^ 2) +
                                                                                     X - 2 * vartotal[["lambda"]]) *
                                                                                    tensorflow::tf$math$exp(-X / vartotal[["lambda"]])) /
                                                                                   ((vartotal[["lambda"]] ^ 2) * (vartotal[["lambda"]] - 1))))
        } else if (dist == "Weibull") {
                loss <- -n * tensorflow::tf$math$log(vartotal[["shape"]]) + vartotal[["shape"]] * n * tensorflow::tf$math$log(vartotal[["scale"]]) -
                        (vartotal[["shape"]] - 1) * tensorflow::tf$reduce_sum(tensorflow::tf$math$log(X)) +
                        tensorflow::tf$reduce_sum((X / vartotal[["scale"]]) ^ vartotal[["shape"]])
        } else if (dist == "DoubleExponential") {
                loss <- -n * tensorflow::tf$math$log(1 / (2 * vartotal[["scale"]])) +
                        (1 / vartotal[["scale"]]) * tensorflow::tf$reduce_sum(tensorflow::tf$abs(X - vartotal[["loc"]]))
        } else if (dist == "Normal") {
                loss <- -(n/2) * tensorflow::tf$math$log(2 * pi) + (n/2) * tensorflow::tf$math$log(vartotal[["sd"]]^2) +
                        (1/(2*vartotal[["sd"]]^2)) * tensorflow::tf$reduce_sum((X - vartotal[["mean"]])^2)
        } else if (dist == "Poisson") {
                loss <- tensorflow::tf$reduce_sum(-X * tensorflow::tf$math$log(vartotal[["lambda"]]) + vartotal[["lambda"]])
        } else if (dist == "Gamma") {
                loss <- -tensorflow::tf$reduce_sum(tensorflow::tf$math$log(((X^(vartotal[["shape"]] - 1)) * tensorflow::tf$math$exp(-vartotal[["rate"]] * X)) /
                                                                            (tensorflow::tf$math$exp(tensorflow::tf$math$lgamma(vartotal[["shape"]])) * vartotal[["rate"]] ^ {-vartotal[["shape"]]})))
        } else if (dist == "Exponential") {
                loss <- -tensorflow::tf$reduce_sum(tensorflow::tf$math$log(vartotal[["rate"]] * tensorflow::tf$math$exp(-vartotal[["rate"]] * X)))
        } else if (dist == "LogNormal") {
                loss <- -tensorflow::tf$reduce_sum(tensorflow::tf$math$log((tensorflow::tf$math$exp(-((tensorflow::tf$math$log(X) - vartotal[["meanlog"]])^2) / (2*vartotal[["sdlog"]]^2)))/
                                                                                   (tensorflow::tf$math$sqrt(2*pi)*vartotal[["sdlog"]]*X)))

        } else if (dist == "Beta") {
                loss <- -tensorflow::tf$reduce_sum(tensorflow::tf$math$log(((X^{vartotal[["shape1"]]-1})*((1-X)^{vartotal[["shape2"]]-1})) /
                                                                                   (tensorflow::tf$math$exp(tensorflow::tf$math$lgamma(vartotal[["shape1"]]))*
                                                                                            (tensorflow::tf$math$exp(tensorflow::tf$math$lgamma(vartotal[["shape2"]]))/
                                                                                                     tensorflow::tf$math$exp(tensorflow::tf$math$lgamma(vartotal[["shape2"]]+ vartotal[["shape1"]]))))))
        }

        return(loss)
}


#------------------------------------------------------------------------
# Hessian Matrix Error --------------------------------------------------
#------------------------------------------------------------------------

hessian_matrix_try <- function(mhess){
        tryCatch(
                expr = {
                        diagvarcov <- sqrt(diag(solve(mhess)))
                        return(diagvarcov)
                },
                error = function(e){
                        message('Caught an error!')
                        print(e)
                        message(paste0('Check the design matrix because it may not be invertible, that is, \n',
                                       'the matrix has linearly dependent columns which means that there are \n',
                                       'strongly correlated variables. This also happens when having more variables \n',
                                       'than observarions and in this case, the design matrix is not full rank.'))
                        return(NULL)
                },
                warning = function(w){
                        message('Caught an warning!')
                        print(w)
                        return(NULL)
                }
        )
}
