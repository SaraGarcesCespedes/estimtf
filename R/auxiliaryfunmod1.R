#' @title disableagerdist function
#'
#' @description Function to estimate distributional parameters disabling the TensorFlow eager execution mode
#'
#' @author Sara Garcés Céspedes
#'
#' @param x a vector with data
#' @param dist an expression indicating the density or mass function depending on xdist
#' @param fixparam a list of the fixed parameters of the distribution of interest. The list must contain the parameters values and names
#' @param initparam a list with initial values of the parameters to be estimated. The list must contain the parameters values and names
#' @param opt an expression indicating the TensorFlow optimizer
#' @param hyperparameters a list with the hyperparameters values of the TensorFlow optimizer
#' @param maxiter a positive integer indicating the maximum number of iterations for the optimization algorithm
#' @param tolerance a small positive number indicating the FALTA FALTA
#' @param np a integer value indicating the number of parameters to be estimated
#'
#' @return
#'
#' @examples
disableagerdist <- function(x, dist, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np) {

        # Disable eager execution
        tf$compat$v1$disable_eager_execution()

        # Create placeholder
        X <- tf$compat$v1$placeholder(dtype=tf$float32, name = "x_data")

        # Create list to store the parameters to be estimated
        var_list <- vector(mode = "list", length = np)

        # Create tf Variables
        #for (i in 1:np) {
         #       var_list[[i]] <- assign(names(initparam)[i], tf$Variable(initparam[[i]], dtype = tf$float32, name = names(initparam)[i]))
        #}

        var_list <- lapply(1:np, FUN = function(i) var_list[[i]] <- assign(names(initparam)[i],
                                                                           tf$Variable(initparam[[i]],
                                                                                       dtype = tf$float32,
                                                                                       name = names(initparam)[i]), envir = .GlobalEnv))
        names(var_list) <- names(initparam)

        # Create a list with all parameters, fixed and not fixed
        vartotal <- append(fixparam, var_list)

        # Create vectors to store parameters, gradientes and loss values of each iteration
        loss <- new_list <- parameters <- gradients <- itergrads <- objvariables <- vector(mode = "list")

        # Create list with variables without names
        #for (i in 1:np) new_list[[i]] <- var_list[[i]]
        new_list <- lapply(1:np, FUN = function(i) new_list[[i]] <- var_list[[i]])

        # Define loss function depending on the distribution
        if (dist == "Poisson") {
                loss_value <- tf$reduce_sum(-X * (tf$math$log(vartotal[["lambda"]])) + vartotal[["lambda"]])
        } else if (dist == "FWE") {
                loss_value <- -tf$reduce_sum(tf$math$log(vartotal[["mu"]] + vartotal[["sigma"]] / (X ^ 2))) - tf$reduce_sum(vartotal[["mu"]] * X - vartotal[["sigma"]] / X) + tf$reduce_sum(tf$math$exp(vartotal[["mu"]] * X - vartotal[["sigma"]] / X))
        } else if (dist == "Instantaneous Failures") {
                loss_value <- -tf$reduce_sum(tf$math$log((((vartotal[["lambda"]] ^ 2) + X - 2 * vartotal[["lambda"]]) * tf$math$exp(-X / vartotal[["lambda"]])) / ((vartotal[["lambda"]] ^ 2) * (vartotal[["lambda"]] - 1))))
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

        maxiter <- 10000
        while(TRUE){
                # Update step
                step <- step + 1

                # Gradient step
                sess$run(train, feed_dict = fd)

                # Parameters and gradients as numeric vectors
                #for (i in 1:np) {
                 #       objvariables[[i]] <- as.numeric(sess$run(new_list[[i]]))
                        #objvariables[[i]] <- as.numeric(objvariables[[i]])
                  #      itergrads[[i]] <- as.numeric(sess$run(grads, feed_dict = fd)[[i]])
                #}

                objvariables <- lapply(1:np, FUN = function(i) objvariables[[i]] <- as.numeric(sess$run(new_list[[i]])))
                itergrads <- lapply(1:np, FUN = function(i) itergrads[[i]] <- as.numeric(sess$run(grads, feed_dict = fd)[[i]]))

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
        #for(i in 1:np) hesslist[[i]] <- tf$gradients(grads[[i]], new_list)
        hesslist <- lapply(1:np, FUN = function(i) hesslist[[i]] <- tf$gradients(grads[[i]], new_list))
        hess <- tf$stack(values=hesslist, axis=0)
        #hess <- tf$reshape(hess, shape(np, np))
        mhess <- sess$run(hess, feed_dict = fd)
        diagvarcov <- sqrt(diag(solve(mhess)))
        #for (i in 1:np) stderror[[i]] <- diagvarcov[i] #ESTO PUEDE SER MAS EFICIENTE
        stderror <- lapply(1:np, FUN = function(i) stderror[[i]] <- diagvarcov[i])
        names(stderror) <- names(var_list)

        # Close tf session
        sess$close()

        # Organize results of each iteration
        gradients <- purrr::transpose(gradients)
        parameters <- purrr::transpose(parameters)
        gradientsfinal <- parametersfinal <- namesgradients <- as.numeric()
        #for (j in 1:np) {
         #       gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[j]]))
          #      parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[j]]))
           #     namesgradients <- cbind(namesgradients, paste0("Gradients ", names(var_list)[j]))
        #}
        gradientsfinal <- sapply(1:np, FUN = function(i) gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[i]])))
        parametersfinal <- sapply(1:np, FUN = function(i) parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[i]])))
        namesgradients <- sapply(1:np, FUN = function(i) namesgradients <- cbind(namesgradients, paste0("Gradients ", names(var_list)[i])))

        # Table of results
        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(var_list), namesgradients)
        return(list(results = results.table, final = tail(results.table, 1), standarderror = stderror))


}


#' @title eagerdist function
#'
#' @description Function to estimate distributional parameters in TensorFlow eager execution mode
#'
#' @author Sara Garcés Céspedes
#'
#' @param x a vector with data
#' @param dist an expression indicating the density or mass function depending on xdist
#' @param fixparam a list of the fixed parameters of the distribution of interest. The list must contain the parameters values and names
#' @param initparam a list with initial values of the parameters to be estimated. The list must contain the parameters values and names
#' @param opt an expression indicating the TensorFlow optimizer
#' @param hyperparameters a list with the hyperparameters values of the TensorFlow optimizer
#' @param maxiter a positive integer indicating the maximum number of iterations for the optimization algorithm
#' @param tolerance a small positive number indicating the FALTA FALTA
#' @param np a integer value indicating the number of parameters to be estimated
#'
#' @return
#'
#' @examples
eagerdist <- function(x, dist, fixparam, linkfun, initparam, opt, hyperparameters, maxiter, tolerance, np) {

        #tf$compat$v1$enable_eager_execution()
        # Create list to store the parameters to be estimated
        var_list <- vector(mode = "list", length = np)

        # Create tf Variables
        #for (i in 1:np) {
         #       var_list[[i]] <- assign(names(initparam)[i], tf$Variable(initparam[[i]], dtype = tf$float32, name = names(initparam)[i]))
        #}
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
        #for (i in 1:np) new_list[[i]] <- var_list[[i]]
        new_list <- lapply(1:np, FUN = function(i) new_list[[i]] <- var_list[[i]])

        maxiter <- 10000

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
                        #for(i in 1:np) hesslist[[i]] <- tape$gradient(grads[[i]], new_list)
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
                #for (i in 1:np) {
                #        objvariables[[i]] <- as.numeric(get(names(var_list)[i]))
                 #       gradients[[step]][[i]] <- as.numeric(gradients[[step]][[i]])
                #}
                objvariables <- lapply(1:np, FUN = function(i) objvariables[[i]] <- as.numeric(get(names(var_list)[i])))
                gradients[[step]] <- lapply(1:np, FUN = function(i) gradients[[step]][[i]] <- as.numeric(gradients[[step]][[i]]))

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
        #for (i in 1:np) stderror[[i]] <- diagvarcov[i] #ESTO PUEDE SER MAS EFICIENTE
        stderror <- lapply(1:np, FUN = function(i) stderror[[i]] <- diagvarcov[i])

        # Organize results of each iteration
        gradients <- purrr::transpose(gradients)
        parameters <- purrr::transpose(parameters)
        gradientsfinal <- parametersfinal <- namesgradients <- as.numeric()
        #for (j in 1:np) {
         #       gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[j]]))
          #      parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[j]]))
           #     namesgradients <- cbind(namesgradients, paste0("Gradients ", names(var_list)[j]))
        #}
        gradientsfinal <- sapply(1:np, function(i) gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[i]])))
        parametersfinal <- sapply(1:np, function(i) parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[i]])))
        namesgradients <- sapply(1:np, function(i) namesgradients <- cbind(namesgradients, paste0("Gradients ", names(var_list)[i])))

        # Table of results
        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(var_list), namesgradients)
        return(list(results = results.table, final = tail(results.table, 1), standarderror = stderror))
}


#' @title comparisondist function
#'
#' @description Function to compare TensorFlow parameter estimations with estimations from other R functions
#'
#' @author Sara Garcés Céspedes
#'
#' @param x a vector with data FALTA FALTA
#' @param xdist a character indicating the name of the distribution of interest. The default value is \code{'Normal'}
#' @param fixparam a list of the fixed parameters of the distribution of interest. The list must contain the parameters values and names
#' @param initparam a list with initial values of the parameters to be estimated. The list must contain the parameters values and names
#' @param lower a numeric vector with lower bounds, with the same lenght of argument `initparam`
#' @param upper a numeric vector with upper bounds, with the same lenght of argument `initparam`
#' @param method a character with the name of the optimization routine. \code{nlminb}, \code{optim}, \code{DEoptim} are available
#'
#' @return
#'
#' @examples
comparisondist <- function(x, xdist, fixparam, initparam, lower, upper, method) {
        distributionsr <- list(Bernoulli = "dbinom", Beta = "dbeta", Exponential = "dexp", Gamma = "dgamma",
                               Normal = "dnorm", Uniform = "dunif", Poisson = "dpois", FWE = "dFWE")

        parametersr <- list(loc = "mean", scale = "sd", concentration1 = "shape1", concentration2 = "shape2",
                            concentration = "shape", low = "min", high = "max", lambda = "lambda", mu = "mu",
                            sigma = "sigma")


        #if (!is.null(fixparam)) for (i in 1:length(fixparam)) names(fixparam)[i] <- parametersr[[match(names(fixparam)[i], names(parametersr))]]
        #if (!is.null(initparam)) for (i in 1:length(initparam)) names(initparam)[i] <- parametersr[[match(names(initparam)[i], names(parametersr))]]

        if (!is.null(fixparam)) names(fixparam) <- lapply(1:length(fixparam), FUN = function(i) names(fixparam)[i] <- parametersr[[match(names(fixparam)[i], names(parametersr))]])
        if (!is.null(initparam)) names(initparam) <- lapply(1:length(initparam), FUN = function(i) names(initparam)[i] <- parametersr[[match(names(initparam)[i], names(parametersr))]])

        estimation <- maxlogL(x = x, dist = distributionsr[[xdist]], fixed = fixparam,
                              start = initparam, optimizer = method, lower = lower,
                              upper = upper)
        return(estimation)


}
