#------------------------------------------------------------------------
# Estimation of distribution parameters (disable eager execution) -------
#------------------------------------------------------------------------

disableagerdist <- function(x, dist, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np, distnotf, xdist, optimizer, lower, upper) {

        # Disable eager execution
        tensorflow::tf$compat$v1$disable_eager_execution()

        # Create placeholder
        X <- tensorflow::tf$compat$v1$placeholder(dtype=tf$float32, name = "x_data")

        # Create list to store the parameters to be estimated
        var_list <- vector(mode = "list", length = np)

        # Create tf Variables
        var_list <- lapply(1:np,
                           FUN = function(i) var_list[[i]] <- assign(names(initparam)[i],
                                                                     tensorflow::tf$Variable(initparam[[i]],
                                                                                             dtype = tf$float32,
                                                                                             name = names(initparam)[i]),
                                                                     envir = .GlobalEnv))

        names(var_list) <- names(initparam)


        # lower and upper limits
        if (!is.null(lower) & !is.null(upper)) {
                var_list_new <- vector(mode = "list", length = np)
                var_list_new <- lapply(1:np,
                                       FUN = function(i) var_list_new[[i]] <- assign(names(initparam)[i],
                                                                                     tf$clip_by_value(get(names(initparam)[i], envir = .GlobalEnv),
                                                                                                      clip_value_min=lower[[names(initparam)[i]]],
                                                                                                      clip_value_max=upper[[names(initparam)[i]]]),
                                                                                     envir = .GlobalEnv))

                names(var_list_new) <- names(initparam)
                var_list <- var_list_new
        } else if (!is.null(lower) & is.null(upper)) {
                upper <- vector(mode = "list", length = np)
                upper <- lapply(1:np, FUN = function(i) upper[[i]] <- Inf)
                names(upper) <- names(var_list)

                var_list_new <- vector(mode = "list", length = np)
                var_list_new <- lapply(1:np,
                                       FUN = function(i) var_list_new[[i]] <- assign(names(initparam)[i],
                                                                                     tf$clip_by_value(get(names(initparam)[i], envir = .GlobalEnv),
                                                                                                      clip_value_min=lower[[names(initparam)[i]]],
                                                                                                      clip_value_max=upper[[names(initparam)[i]]]),
                                                                                     envir = .GlobalEnv))

                names(var_list_new) <- names(initparam)
                var_list <- var_list_new
        } else if (is.null(lower) & !is.null(upper)) {
                lower <- vector(mode = "list", length = np)
                lower <- lapply(1:np, FUN = function(i) lower[[i]] <- -Inf)
                names(lower) <- names(var_list)

                var_list_new <- vector(mode = "list", length = np)
                var_list_new <- lapply(1:np,
                                       FUN = function(i) var_list_new[[i]] <- assign(names(initparam)[i],
                                                                                     tf$clip_by_value(get(names(initparam)[i], envir = .GlobalEnv),
                                                                                                      clip_value_min=lower[[names(initparam)[i]]],
                                                                                                      clip_value_max=upper[[names(initparam)[i]]]),
                                                                                     envir = .GlobalEnv))

                names(var_list_new) <- names(initparam)
                var_list <- var_list_new
        }



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
        hesslist <- lapply(1:np, FUN = function(i) hesslist[[i]] <- tensorflow::tf$gradients(grads[[i]], new_list))
        hess <- tensorflow::tf$stack(values=hesslist, axis=0)
        mhess <- sess$run(hess, feed_dict = fd)
        diagvarcov <- hessian_matrix_try(mhess)
        #diagvarcov <- sqrt(diag(solve(mhess)))
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

        # change name of parameters to match R parameters
        names_param <- names(initparam)
        names_new <- vector(mode = "numeric", length = length(names_param))
        names_new <- sapply(1:length(names_param), FUN = function(i) names_new[i] <- parameter_name_R(names_param[i], xdist))
        names(initparam) <- names_new

        # Table of results
        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(var_list), namesgradients)
        outputs <- list(n = n, type = "MLEdistf", parnames = names(initparam),
                        estimates = tail(results.table[, 2:(np + 1)], 1),
                        convergence = convergence)
        result <- list(results = results.table, vcov = mhess, standarderror = stderror,
                       outputs = outputs)
        return(result)
}

#------------------------------------------------------------------------
# Loss function for distributions not included in TF --------------------
#------------------------------------------------------------------------
lossfun <- function(dist, vartotal, X) {
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
                },
                warning = function(w){
                        message('Caught an warning!')
                        print(w)
                }
        )
}
