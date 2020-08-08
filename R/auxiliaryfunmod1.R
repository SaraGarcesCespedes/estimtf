#------------------------------------------------------------------------
# Estimation of distribution parameters (disable eager execution) -------
#------------------------------------------------------------------------

disableagerdist <- function(x, dist, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np, distnotf, xdist, optimizer) {

        # Disable eager execution
        tf$compat$v1$disable_eager_execution()

        # Create placeholder
        X <- tf$compat$v1$placeholder(dtype=tf$float32, name = "x_data")

        # Create list to store the parameters to be estimated
        var_list <- vector(mode = "list", length = np)

        # Create tf Variables
        var_list <- lapply(1:np,
                           FUN = function(i) var_list[[i]] <- assign(names(initparam)[i],
                                                                           tf$Variable(initparam[[i]],
                                                                                       dtype = tf$float32,
                                                                                       name = names(initparam)[i]),
                                                                     envir = .GlobalEnv))

        names(var_list) <- names(initparam)

        # Create a list with all parameters, fixed and not fixed
        vartotal <- append(fixparam, var_list)

        # Create vectors to store parameters, gradientes and loss values of each iteration
        loss <- new_list <- parameters <- gradients <- itergrads <- objvariables <- vector(mode = "list")

        # Create list with variables without names
        new_list <- lapply(1:np, FUN = function(i) new_list[[i]] <- var_list[[i]])

        n <- length(x)
        # Define loss function depending on the distribution
        if (xdist %in% distnotf) {
                loss_value <- lossfun(dist, vartotal, X)
        } else {
                density <- do.call(what = dist, vartotal)
                loss_value <- tf$negative(tf$reduce_sum(density$log_prob(value = X)))
        }

        # Compute gradients
        grads <- tf$gradients(loss_value, new_list)

        # Define optimizer
        if (optimizer == "GradientDescentOptimizer") {
                global_step <- tf$Variable(0, trainable = FALSE)
                starter_learning_rate <- hyperparameters$learning_rate
                learning.rate <- tf$compat$v1$train$exponential_decay(starter_learning_rate, global_step,
                                                                      100000, 0.96, staircase=TRUE)
                hyperparameters$learning_rate <- learning.rate
                seloptimizer <- do.call(what = opt, hyperparameters)
                train <- eval(parse(text = "seloptimizer$minimize(loss_value, global_step = global_step)"))

        } else {
                seloptimizer <- do.call(what = opt, hyperparameters)
                train <- eval(parse(text = "seloptimizer$minimize(loss_value)"))
        }

        # Initialize the variables and open the session
        init <- tf$compat$v1$initialize_all_variables()
        sess <- tf$compat$v1$Session()
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

                # Conditions
                if (step != 1) {
                        if (abs(loss[[step]] - loss[[step-1]]) < tolerance$loss){
                                convergence <- paste("Loss function convergence,", step, "iterations needed.")
                                break
                        } else if (step >= maxiter) {
                                convergence <- paste("Maximum number of iterations reached.")
                                break
                        } else if (isTRUE(sapply(1:np, FUN= function(x) abs(parameters[[step]][[x]]) < tolerance$parameters))) {
                                convergence <- paste("Parameters convergence,", step, "iterations needed.")
                                break
                        } else if (isTRUE(sapply(1:np, FUN= function(x) abs(gradients[[step]][[x]]) < tolerance$gradients))) {
                                convergence <- paste("Gradients convergence,", step, "iterations needed.")
                                break
                        }
                }
        }

        # Compute Hessian matrix
        hesslist <- stderror <- vector(mode = "list", length = np)
        hesslist <- lapply(1:np, FUN = function(i) hesslist[[i]] <- tf$gradients(grads[[i]], new_list))
        hess <- tf$stack(values=hesslist, axis=0)
        mhess <- sess$run(hess, feed_dict = fd)
        diagvarcov <- sqrt(diag(solve(mhess)))
        stderror <- lapply(1:np, FUN = function(i) stderror[[i]] <- diagvarcov[i])
        names(stderror) <- names(var_list)

        # Close tf session
        sess$close()

        # Organize results of each iteration
        gradients <- purrr::transpose(gradients)
        parameters <- purrr::transpose(parameters)
        gradientsfinal <- parametersfinal <- namesgradients <- as.numeric()
        gradientsfinal <- sapply(1:np, FUN = function(i) gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[i]])))
        parametersfinal <- sapply(1:np, FUN = function(i) parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[i]])))
        namesgradients <- sapply(1:np, FUN = function(i) namesgradients <- cbind(namesgradients, paste0("Gradients ", names(var_list)[i])))

        # Table of results
        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(var_list), namesgradients)
        outputs <- list(n = n, type = "MLEdistf", parnames = names(initparam),
                        estimates = tail(results.table[, 2:(np + 1)], 1),
                        convergence = convergence)
        result <- list(results = results.table, vcov = diagvarcov, standarderror = stderror,
                       outputs = outputs)
        return(result)
}

#------------------------------------------------------------------------
# Estimation of distribution parameters (with eager execution) ----------
#------------------------------------------------------------------------

eagerdist <- function(x, dist, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np, distnotf, xdist, optimizer) {

        # Create list to store the parameters to be estimated
        var_list <- vector(mode = "list", length = np)

        # Create tf Variables
        var_list <- lapply(1:np, FUN = function(i) var_list[[i]] <- assign(names(initparam)[i],
                                                                           tf$Variable(initparam[[i]],
                                                                                       dtype = tf$float32,
                                                                                       name = names(initparam)[i]),
                                                                           envir = .GlobalEnv))
        names(var_list) <- names(initparam)

        # Create a list with all parameters, fixed and not fixed
        vartotal <- append(fixparam, var_list)

        # Define optimizer
        seloptimizer <- do.call(what = opt, hyperparameters)

        # Initialize step
        step <- 0

        # Create vectors to store parameters, gradientes and loss values of each iteration
        loss <- new_list <- parameters <- gradients <- hesslist <- objvariables <- vector(mode = "list")

        # Create list with variables without names
        new_list <- lapply(1:np, FUN = function(i) new_list[[i]] <- var_list[[i]])

        while(TRUE){
                # Update step
                step <- step + 1

                with(tf$GradientTape(persistent = TRUE) %as% tape, {
                        # Define loss function depending on the distribution
                        X <- x
                        n <- length(x)
                        if (xdist %in% distnotf) {
                                loss_value <- lossfun(dist, vartotal, X)
                        } else {
                                density <- do.call(what = dist, vartotal)
                                loss_value <- tf$negative(tf$reduce_sum(density$log_prob(value = X)))
                        }
                        grads <- tape$gradient(loss_value, new_list)
                        # Compute Hessian matrixin
                        hesslist <- lapply(1:np, FUN = function(i) hesslist[[i]] <- tape$gradient(grads[[i]], new_list))
                        mhess <- as.matrix(tf$stack(values=hesslist, axis=0))
                })

                # Compute gradientes
                seloptimizer$apply_gradients(purrr::transpose(list(grads, new_list)))

                # Save loss value
                loss[[step]] <- as.numeric(loss_value)

                # Save gradients values
                gradients[[step]] <- grads

                # Parameters and gradients as numeric vectors
                objvariables <- lapply(1:np, FUN = function(i) objvariables[[i]] <- as.numeric(get(names(var_list)[i])))
                gradients[[step]] <- lapply(1:np, FUN = function(i) gradients[[step]][[i]] <- as.numeric(gradients[[step]][[i]]))

                parameters[[step]] <- objvariables

                # Conditions
                if (step != 1) {
                        if (abs(loss[[step]] - loss[[step-1]]) < tolerance$loss){
                                convergence <- paste("Loss function convergence,", step, "iterations needed.")
                                break
                        } else if (step >= maxiter) {
                                convergence <- paste("Maximum number of iterations reached.")
                                break
                        } else if (isTRUE(sapply(1:np, FUN= function(x) abs(parameters[[step]][[x]]) < tolerance$parameters))) {
                                convergence <- paste("Parameters convergence,", step, "iterations needed.")
                                break
                        } else if (isTRUE(sapply(1:np, FUN= function(x) abs(gradients[[step]][[x]]) < tolerance$gradients))) {
                                convergence <- paste("Gradients convergence,", step, "iterations needed.")
                                break
                        }
                }
        }

        # Compute std error for each estimator
        stderror <- vector(mode = "list", length = np)
        diagvarcov <- sqrt(diag(solve(mhess)))
        names(stderror) <- names(var_list)
        stderror <- lapply(1:np, FUN = function(i) stderror[[i]] <- diagvarcov[i])

        # Organize results of each iteration
        gradients <- purrr::transpose(gradients)
        parameters <- purrr::transpose(parameters)
        gradientsfinal <- parametersfinal <- namesgradients <- as.numeric()
        gradientsfinal <- sapply(1:np, function(i) gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[i]])))
        parametersfinal <- sapply(1:np, function(i) parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[i]])))
        namesgradients <- sapply(1:np, function(i) namesgradients <- cbind(namesgradients, paste0("Gradients ", names(var_list)[i])))

        # Table of results
        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(var_list), namesgradients)
        outputs <- list(n = n, type = "MLEdistf", parnames = names(initparam),
                        estimates= tail(results.table[, 2:(np + 1)], 1),
                        convergence = convergence)
        result <- list(results = results.table, vcov = diagvarcov, standarderror = stderror,
                       outputs = outputs)
        return(result)
}

#------------------------------------------------------------------------
# Loss function for distributions not included in TF --------------------
#------------------------------------------------------------------------
lossfun <- function(dist, vartotal, X) {
        if (dist == "Poisson") {
                loss <- tf$reduce_sum(-X * (tf$math$log(vartotal[["lambda"]])) + vartotal[["lambda"]])
        } else if (dist == "FWE") {
                loss <- -tf$reduce_sum(tf$math$log(vartotal[["mu"]] + vartotal[["sigma"]] / (X ^ 2))) -
                        tf$reduce_sum(vartotal[["mu"]] * X - vartotal[["sigma"]] / X) +
                        tf$reduce_sum(tf$math$exp(vartotal[["mu"]] * X - vartotal[["sigma"]] / X))
        } else if (dist == "InstantaneousFailures") {
                loss <- -tf$reduce_sum(tf$math$log((((vartotal[["lambda"]] ^ 2) +
                                                             X - 2 * vartotal[["lambda"]]) *
                                                            tf$math$exp(-X / vartotal[["lambda"]])) /
                                                           ((vartotal[["lambda"]] ^ 2) * (vartotal[["lambda"]] - 1))))
        } else if (dist == "Weibull") {
                loss <- -n * tf$math$log(vartotal[["shape"]]) + vartotal[["shape"]] * n * tf$math$log(vartotal[["scale"]]) -
                        (vartotal[["shape"]] - 1) * tf$reduce_sum(tf$math$log(X)) +
                        tf$reduce_sum((X / vartotal[["scale"]]) ^ vartotal[["shape"]])
        } else if (dist == "Cauchy") {
                loss <- n * tf$math$log(pi * vartotal[["scale"]]) +
                        tf$reduce_sum(tf$math$log(1 + ((X - vartotal[["loc"]]) / vartotal[["scale"]])^2))

        } else if (dist == "Geometric") {
                loss <- -n * tf$math$log(vartotal[["prob"]]) -
                        (tf$reduce_sum(X) - n) * tf$math$log(1 - vartotal[["prob"]])
        } else if (dist == "DoubleExponential") {
                loss <- -n * tf$math$log(1 / (2 * vartotal[["scale"]])) +
                        (1 / vartotal[["scale"]]) * tf$reduce_sum(tf$abs(X - vartotal[["loc"]]))
        } else if (dist == "LogNormal") {
                loss <- (n / 2) * tf$math$log(2 * pi * vartotal[["sdlog"]] ^ 2) +
                        tf$reduce_sum(tf$math$log(X)) +
                        (tf$reduce_sum(tf$math$log(X ^ 2)) / (2 * vartotal[["sdlog"]] ^ 2)) -
                        (tf$reduce_sum(tf$math$log(X) * vartotal[["meanlog"]]) /  (vartotal[["sdlog"]] ^ 2)) +
                        ((n *  vartotal[["meanlog"]] ^ 2) / (2 * vartotal[["sdlog"]] ^ 2))
        }

        return(loss)
}
