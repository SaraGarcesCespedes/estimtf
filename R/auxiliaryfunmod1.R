#' @title disableager_estimtf function
#'
#' @description Function to estimate distributional parameters disabling the TensorFlow eager execution mode
#'
#' @param x
#' @param dist
#' @param fixparam
#' @param linkfun
#' @param initparam
#' @param opt
#' @param hyperparameters
#' @param maxiter
#' @param tolerance
#' @param np
#'
#' @return
#'
#' @examples
disableager_estimtf <- function(x, dist, fixparam, linkfun, initparam, opt, hyperparameters, maxiter, tolerance, np) {

        # Disable eager execution
        tf$compat$v1$disable_eager_execution()

        # Create placeholder
        X <- tf$compat$v1$placeholder(dtype=tf$float32, name = "x_data")

        # Create list to store the parameters to be estimated
        var_list <- vector(mode = "list", length = np)
        names(var_list) <- names(initparam)

        # Create tf Variables
        for (i in 1:np) {
                var_list[[i]] <- assign(names(initparam)[i], tf$Variable(initparam[[i]], dtype = tf$float32, name = names(initparam)[i]))
        }

        # Create a list with all parameters, fixed and not fixed
        vartotal <- append(fixparam, var_list)

        # Create vectors to store parameters, gradientes and loss values of each iteration
        loss <- new_list <- parameters <- gradients <- itergrads <- objvariables <- vector(mode = "list")

        # Create list with variables without names
        for (i in 1:np) new_list[[i]] <- var_list[[i]]

        # Define loss function depending on the distribution
        if (dist == "Poisson") {
                loss_value <- tf$reduce_sum(-x * (tf$math$log(vartotal[["lambda"]])) + vartotal[["lambda"]])
        } else if (dist == "FWE") {
                loss_value <- -tf$reduce_sum(tf$math$log(vartotal[["mu"]] + vartotal[["sigma"]] / (x ^ 2))) - tf$reduce_sum(vartotal[["mu"]] * x - vartotal[["sigma"]] / x) + tf$reduce_sum(tf$math$exp(vartotal[["mu"]] * x - vartotal[["sigma"]] / x))
        } else if (dist == "Instantaneous Failures") {
                loss_value <- -tf$reduce_sum(tf$math$log((((vartotal[["lambda"]] ^ 2) + x - 2 * vartotal[["lambda"]]) * tf$math$exp(-x / vartotal[["lambda"]])) / ((vartotal[["lambda"]] ^ 2) * (vartotal[["lambda"]] - 1))))
        } else {
                density <- do.call(what = dist, vartotal)
                loss_value <- tf$negative(tf$reduce_sum(density$log_prob(value = X)))
        }

        # Compute gradients
        grads <- tf$gradients(loss_value, new_list)

        # Define optimizer
        seloptimizer <- do.call(what = opt, hyperparameters)
        train <- eval(parse(text = "seloptimizer$minimize(loss_value)"))

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

                # Parameters and gradients as numeric vectors
                for (i in 1:np) {
                        objvariables[[i]] <- as.numeric(sess$run(new_list[[i]]))
                        #objvariables[[i]] <- as.numeric(objvariables[[i]])
                        itergrads[[i]] <- as.numeric(sess$run(grads, feed_dict = fd)[[i]])
                }

                parameters[[step]] <- objvariables
                gradients[[step]] <- itergrads

                # Save loss value
                loss[[step]] <- as.numeric(sess$run(loss_value, feed_dict = fd))

                # Conditions
                if (step != 1) {
                        if (abs(loss[[step]] - loss[[step-1]]) < tolerance$loss){
                                print(paste("Loss function convergence,", step, "iterations needed."))
                                break
                        } else if (step >= maxiter) {
                                print(paste("Maximum number of iterations reached."))
                                break
                        } else if (isTRUE(sapply(1:np, FUN= function(x) abs(parameters[[step]][[x]]) < tolerance$parameters))) {
                                print(paste("Parameters convergence,", step, "iterations needed."))
                                break
                        } else if (isTRUE(sapply(1:np, FUN= function(x) abs(gradients[[step]][[x]]) < tolerance$gradients))) {
                                print(paste("Gradients convergence,", step, "iterations needed."))
                                break
                        }
                }
        }

        # Compute Hessian matrix
        hesslist <- stderror <- vector(mode = "list", length = np)
        for(i in 1:np) hesslist[[i]] <- tf$gradients(grads[[i]], new_list)
        hess <- tf$stack(values=hesslist, axis=0)
        #hess <- tf$reshape(hess, shape(np, np))
        mhess <- sess$run(hess, feed_dict = fd)
        diagvarcov <- sqrt(diag(solve(mhess)))
        names(stderror) <- names(var_list)
        for (i in 1:np) stderror[[i]] <- diagvarcov[i] #ESTO PUEDE SER MAS EFICIENTE

        # Close tf session
        sess$close()

        # Organize results of each iteration
        gradients <- purrr::transpose(gradients)
        parameters <- purrr::transpose(parameters)
        gradientsfinal <- parametersfinal <- namesgradients <- as.numeric()
        for (j in 1:np) {
                gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[j]]))
                parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[j]]))
                namesgradients <- cbind(namesgradients, paste0("Gradients ", names(var_list)[j]))
        }
        # Table of results
        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(var_list), namesgradients)
        return(list(results = results.table, final = tail(results.table, 1), standarderror = stderror))


}


#' @title eager_estimtf function
#'
#' @description Function to estimate distributional parameters in TensorFlow eager execution mode
#'
#' @param x
#' @param dist
#' @param fixparam
#' @param linkfun
#' @param initparam
#' @param opt
#' @param hyperparameters
#' @param maxiter
#' @param tolerance
#' @param np
#'
#' @return
#'
#' @examples
eager_estimtf <- function(x, dist, fixparam, linkfun, initparam, opt, hyperparameters, maxiter, tolerance, np) {

        # Create list to store the parameters to be estimated
        var_list <- vector(mode = "list", length = np)
        names(var_list) <- names(initparam)

        # Create tf Variables
        for (i in 1:np) {
                var_list[[i]] <- assign(names(initparam)[i], tf$Variable(initparam[[i]], dtype = tf$float32, name = names(initparam)[i]))
        }

        # Create a list with all parameters, fixed and not fixed
        vartotal <- append(fixparam, var_list)

        # Define optimizer
        seloptimizer <- do.call(what = opt, hyperparameters)

        # Initialize step
        step <- 0

        # Create vectors to store parameters, gradientes and loss values of each iteration
        loss <- new_list <- parameters <- gradients <- hesslist <- objvariables <- vector(mode = "list")

        # Create list with variables without names
        for (i in 1:np) new_list[[i]] <- var_list[[i]]


        while(TRUE){
                # Update step
                step <- step + 1

                with(tf$GradientTape(persistent = TRUE) %as% tape, {
                        # Define loss function depending on the distribution
                        if (dist == "Poisson") {
                                loss_value <- tf$reduce_sum(-x * (tf$math$log(vartotal[["lambda"]])) + vartotal[["lambda"]])
                        } else if (dist == "FWE") {
                                loss_value <- -tf$reduce_sum(tf$math$log(vartotal[["mu"]] + vartotal[["sigma"]] / (x ^ 2))) - tf$reduce_sum(vartotal[["mu"]] * x - vartotal[["sigma"]] / x) + tf$reduce_sum(tf$math$exp(vartotal[["mu"]] * x - vartotal[["sigma"]] / x))
                        } else if (dist == "Instantaneous Failures") {
                                loss_value <- -tf$reduce_sum(tf$math$log((((vartotal[["lambda"]] ^ 2) + x - 2 * vartotal[["lambda"]]) * tf$math$exp(-x / vartotal[["lambda"]])) / ((vartotal[["lambda"]] ^ 2) * (vartotal[["lambda"]] - 1))))
                        } else {
                                density <- do.call(what = dist, vartotal)
                                loss_value <- tf$negative(tf$reduce_sum(density$log_prob(value = x)))
                        }
                        grads <- tape$gradient(loss_value, new_list)
                        # Compute Hessian matrixin
                        for(i in 1:np) hesslist[[i]] <- tape$gradient(grads[[i]], new_list)
                        mhess <- as.matrix(tf$stack(values=hesslist, axis=0))
                })

                # Compute gradientes
                seloptimizer$apply_gradients(purrr::transpose(list(grads, new_list)))

                # Save loss value
                loss[[step]] <- as.numeric(loss_value)

                # Save gradients values
                gradients[[step]] <- grads

                # Parameters and gradients as numeric vectors
                for (i in 1:np) {
                        objvariables[[i]] <- as.numeric(get(names(var_list)[i]))
                        gradients[[step]][[i]] <- as.numeric(gradients[[step]][[i]])
                }
                parameters[[step]] <- objvariables

                # Conditions
                if (step != 1) {
                        if (abs(loss[[step]] - loss[[step-1]]) < tolerance$loss){
                                print(paste("Loss function convergence,", step, "iterations needed."))
                                break
                        } else if (step >= maxiter) {
                                print(paste("Maximum number of iterations reached."))
                                break
                        } else if (isTRUE(sapply(1:np, FUN= function(x) abs(parameters[[step]][[x]]) < tolerance$parameters))) {
                                print(paste("Parameters convergence,", step, "iterations needed."))
                                break
                        } else if (isTRUE(sapply(1:np, FUN= function(x) abs(gradients[[step]][[x]]) < tolerance$gradients))) {
                                print(paste("Gradients convergence,", step, "iterations needed."))
                                break
                        }
                }
        }

        # Compute std error for each estimator
        stderror <- vector(mode = "list", length = np)
        diagvarcov <- sqrt(diag(solve(mhess)))
        names(stderror) <- names(var_list)
        for (i in 1:np) stderror[[i]] <- diagvarcov[i] #ESTO PUEDE SER MAS EFICIENTE

        # Organize results of each iteration
        gradients <- purrr::transpose(gradients)
        parameters <- purrr::transpose(parameters)
        gradientsfinal <- parametersfinal <- namesgradients <- as.numeric()
        for (j in 1:np) {
                gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[j]]))
                parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[j]]))
                namesgradients <- cbind(namesgradients, paste0("Gradients ", names(var_list)[j]))
        }
        # Table of results
        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(var_list), namesgradients)
        return(list(results = results.table, final = tail(results.table, 1), standarderror = stderror))
}


#' @title comparison_estimtf function
#'
#' @description Function to compare TensorFlow parameter estimations with estimations from other R functions
#'
#' @param x
#' @param xdist
#' @param fixparam
#' @param initparam
#' @param lower
#' @param upper
#' @param method
#'
#' @return
#'
#' @examples
comparison_estimtf <- function(x, xdist, fixparam, initparam, lower, upper, method) {
        distributionsr <- list(Bernoulli = "dbinom", Beta = "dbeta", Exponential = "dexp", Gamma = "dgamma",
                               Normal = "dnorm", Uniform = "dunif")

        parametersr <- list(loc = "mean", scale = "sd", concentration1 = "shape1", concentration2 = "shape2",
                            concentration = "shape", low = "min", high = "max")


        if (!is.null(fixparam)) for (i in 1:length(fixparam)) names(fixparam)[i] <- parametersr[[match(names(fixparam)[i], names(parametersr))]]
        if (!is.null(initparam)) for (i in 1:length(initparam)) names(initparam)[i] <- parametersr[[match(names(initparam)[i], names(parametersr))]]

        estimation <- maxlogL(x = x, dist = distributionsr[[xdist]], fixed = fixparam,
                              start = initparam, optimizer = method)
        return(estimation)


}
