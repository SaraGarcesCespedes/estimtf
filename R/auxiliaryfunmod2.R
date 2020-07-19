#' @title disableagerreg function
#'
#' @description Function to estimate regression parameters disabling the TensorFlow eager execution mode
#'
#' @author Sara Garcés Céspedes
#' @param x
#' @param dist
#' @param design_matrix
#' @param fixparam
#' @param initparam
#' @param opt
#' @param hyperparameters
#' @param maxiter
#' @param tolerance
#' @param np
#'
#' @return
#' @export
#'
#' @examples
disableagerreg <- function(data, dist, design_matrix, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np, link_function, ydist) {

        # Disable eager execution
        tf$compat$v1$disable_eager_execution()

        # Create placeholders
        Y <- tf$compat$v1$placeholder(dtype = tf$float32, name = "y_data")
        y_data <- design_matrix$y

        nbetas <- param <- vector(mode = "list", length = np)
        names(nbetas) <- names(param) <- names(design_matrix)[1:np]
        totalbetas <- sum(as.numeric(unlist(sapply(design_matrix[1:np], ncol))))
        regparam <- vector(mode = "list", length = totalbetas)
        t<- 0
        for (i in 1:np){
                sum <- 0
                nbetas[[i]] <- sum(as.numeric(unlist(sapply(design_matrix[i], ncol))))
                bnum <- rep(0:(nbetas[[i]]-1))
                multparam <- vector(mode = "list", length = nbetas[[i]])
                x_data <- eval(parse(text = paste("design_matrix$", names(nbetas)[i], sep = "")))
                for (j in 1:nbetas[[i]]){
                regparam[[j + t]] <- assign(paste0("beta", bnum[j], names(nbetas)[i]),
                                                      tf$Variable(initparam[[names(nbetas)[i]]],
                                                                  dtype = tf$float32))
                names(regparam)[j + t] <- paste0("beta", bnum[j], names(nbetas)[i])
                multparam[[j]] <- tf$multiply(x_data[, j], regparam[[j + t]])
                sum <- sum + multparam[[j]]
                }
                sum <- link(link_function, sum, names(nbetas)[i])
                param[[i]] <- assign(names(nbetas)[i], sum)
                t <- t + nbetas[[i]]
        }


        # Create a list with all parameters, fixed and not fixed
        vartotal <- append(fixparam, param)

        # Create vectors to store parameters, gradientes and loss values of each iteration
        loss <- new_list <- parameters <- gradients <- itergrads <- objvariables <- vector(mode = "list")

        if (dist == "Poisson") {
                loss_value <- tf$reduce_sum(-Y * (tf$math$log(vartotal[["lambda"]])) + vartotal[["lambda"]])
        } else if (dist == "FWE") {
                loss_value <- -tf$reduce_sum(tf$math$log(vartotal[["mu"]] + vartotal[["sigma"]] / (Y ^ 2))) - tf$reduce_sum(vartotal[["mu"]] * Y - vartotal[["sigma"]] / Y) + tf$reduce_sum(tf$math$exp(vartotal[["mu"]] * Y - vartotal[["sigma"]] / Y))
        } else if (dist == "Instantaneous Failures") {
                loss_value <- -tf$reduce_sum(tf$math$log((((vartotal[["lambda"]] ^ 2) + Y - 2 * vartotal[["lambda"]]) * tf$math$exp(-Y / vartotal[["lambda"]])) / ((vartotal[["lambda"]] ^ 2) * (vartotal[["lambda"]] - 1))))
        } else {
                density <- do.call(what = dist, vartotal)
                loss_value <- tf$negative(tf$reduce_sum(density$log_prob(value = Y)))
        }

        # Compute gradients
        grads <- tf$gradients(loss_value, regparam)

        # Define optimizer
        seloptimizer <- do.call(what = opt, hyperparameters)
        train <- eval(parse(text = "seloptimizer$minimize(loss_value)"))

        # Initialize the variables and open the session
        init <- tf$compat$v1$initialize_all_variables()
        sess <- tf$compat$v1$Session()
        sess$run(init)

        # Create dictionary to feed data into graph
        fd <- dict(Y = y_data)

        # Initialize step
        step <- 0
        maxiter <- 10000

        while(TRUE){
                # Update step
                step <- step + 1

                # Gradient step
                sess$run(train, feed_dict = fd)

                # Parameters and gradients as numeric vectors
                for (i in 1:length(regparam)) {
                        objvariables[[i]] <- as.numeric(sess$run(regparam[[i]]))
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
                        } else if (isTRUE(sapply(1:length(regparam), FUN= function(x) abs(parameters[[step]][[x]]) < tolerance$parameters))) {
                                print(paste("Parameters convergence,", step, "iterations needed."))
                                break
                        } else if (isTRUE(sapply(1:length(regparam), FUN= function(x) abs(gradients[[step]][[x]]) < tolerance$gradients))) {
                                print(paste("Gradients convergence,", step, "iterations needed."))
                                break
                        }
                }
        }

        # TODAVIA FALTAAAAA ARREGLAR ESTO
        # Compute Hessian matrix
        hesslist <- stderror <- vector(mode = "list", length = length(regparam))
        for(i in 1:length(regparam)) hesslist[[i]] <- tf$gradients(grads[[i]], regparam)
        hess <- tf$stack(values=hesslist, axis=0)
        #hess <- tf$reshape(hess, shape(np, np))
        mhess <- sess$run(hess, feed_dict = fd)
        diagvarcov <- sqrt(diag(solve(mhess)))
        #names(stderror) <- names(var_list)
        for (i in 1:length(regparam)) stderror[[i]] <- diagvarcov[i] #ESTO PUEDE SER MAS EFICIENTE

        # Close tf session
        sess$close()

        # Organize results of each iteration
        gradients <- purrr::transpose(gradients)
        parameters <- purrr::transpose(parameters)
        gradientsfinal <- parametersfinal <- namesgradients <- as.numeric()
        for (j in 1:length(regparam)) {
                gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[j]]))
                parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[j]]))
                namesgradients <- cbind(namesgradients, paste0("Gradients ", names(regparam)[j]))
        }
        # Table of results
        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(var_list), namesgradients)
        return(list(results = results.table, final = tail(results.table, 1), standarderror = stderror))


}


#' @title eagerreg function
#'
#' @description Function to estimate regression parameters in TensorFlow eager execution mode
#'
#' @author Sara Garcés Céspedes
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
#' @export
#'
#' @examples
#'
eagerreg <- function(data, dist, design_matrix, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np, link_function, ydist) {

        #y_data <- tf$constant(design_matrix$y, dtype = tf$float32)
        y_data <- as.double(design_matrix$y)

        nbetas <- param <- vector(mode = "list", length = np)
        names(nbetas) <- names(param) <- names(design_matrix)[1:np]
        totalbetas <- sum(as.numeric(unlist(sapply(design_matrix[1:np], ncol))))
        regparam <- vector(mode = "list", length = totalbetas)
        t <- 0
        for (i in 1:np){
                sum <- 0
                nbetas[[i]] <- sum(as.numeric(unlist(sapply(design_matrix[i], ncol))))
                bnum <- rep(0:(nbetas[[i]]-1))
                for (j in 1:nbetas[[i]]){
                        regparam[[j + t]] <- assign(paste0("beta", bnum[j], names(nbetas)[i]),
                                                    tf$Variable(initparam[[names(nbetas)[i]]],
                                                                dtype = tf$float32,
                                                                name = paste0("beta", bnum[j], names(nbetas)[i])))
                        names(regparam)[j + t] <- paste0("beta", bnum[j], names(nbetas)[i])

                }
                t <- t + nbetas[[i]]
        }

        # SI ES NECESARIO?
        if (!is.null(fixparam)) {
                for (j in length(fixparam)) {
                        fixparam[[j]] <- tf$constant(fixparam[[j]], dtype = tf$float32)
                }
        }

        # Create a list with all parameters, fixed and not fixed
        #vartotal <- append(fixparam, param)

        # Define optimizer
        seloptimizer <- do.call(what = opt, hyperparameters)

        # Initialize step
        step <- 0

        # Create vectors to store parameters, gradientes and loss values of each iteration
        loss <- new_list <- parameters <- gradients <- hesslist <- objvariables <- vector(mode = "list")
        for (i in 1:length(regparam)) new_list[[i]] <- regparam[[i]]

        maxiter <- 10000

        while(TRUE){
                # Update step
                step <- step + 1

                with(tf$GradientTape(persistent = TRUE) %as% tape, {
                        t <- 0
                        for (i in 1:np){
                                sum <- 0
                                multparam <- vector(mode = "list", length = nbetas[[i]])
                                x_data <- eval(parse(text = paste("design_matrix$", names(nbetas)[i], sep = "")))
                                for (j in 1:nbetas[[i]]){
                                        multparam[[j]] <- tf$multiply(x_data[, j], regparam[[j + t]])
                                        sum <- sum + multparam[[j]]
                                }
                                sum <- link(link_function, sum, names(nbetas)[i])
                                param[[i]] <- assign(names(nbetas)[i], sum)
                                t <- t + nbetas[[i]]
                        }
                        vartotal <- append(fixparam, param)

                        # Define loss function depending on the distribution
                        if (dist == "Poisson") {
                                loss_value <- tf$reduce_sum(-y_data * (tf$math$log(vartotal[["lambda"]])) + vartotal[["lambda"]])
                        } else if (dist == "FWE") {
                                loss_value <- -tf$reduce_sum(tf$math$log(vartotal[["mu"]] + vartotal[["sigma"]] / (y_data ^ 2))) - tf$reduce_sum(vartotal[["mu"]] * y_data - vartotal[["sigma"]] / y_data) + tf$reduce_sum(tf$math$exp(vartotal[["mu"]] * y_data - vartotal[["sigma"]] / y_data))
                        } else if (dist == "Instantaneous Failures") {
                                loss_value <- -tf$reduce_sum(tf$math$log((((vartotal[["lambda"]] ^ 2) + y_data - 2 * vartotal[["lambda"]]) * tf$math$exp(-y_data / vartotal[["lambda"]])) / ((vartotal[["lambda"]] ^ 2) * (vartotal[["lambda"]] - 1))))
                        } else {
                                density <- do.call(what = dist, vartotal)
                                loss_value <- tf$negative(tf$reduce_sum(density$log_prob(value = y_data)))
                        }
                        grads <- tape$gradient(loss_value, new_list)
                        # Compute Hessian matrixin
                        for(i in 1:length(new_list)) hesslist[[i]] <- tape$gradient(grads[[i]], new_list)
                        mhess <- as.matrix(tf$stack(values=hesslist, axis=0))
                })


                # Compute gradientes
                seloptimizer$apply_gradients(purrr::transpose(list(grads, new_list)))

                # Save loss value
                loss[[step]] <- as.numeric(loss_value)

                # Save gradients values
                gradients[[step]] <- grads

                # Parameters and gradients as numeric vectors
                for (i in 1:length(regparam)) {
                        objvariables[[i]] <- as.numeric(get(names(regparam)[i]))
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
                        } else if (isTRUE(sapply(1:length(regparam), FUN= function(x) abs(parameters[[step]][[x]]) < tolerance$parameters))) {
                                print(paste("Parameters convergence,", step, "iterations needed."))
                                break
                        } else if (isTRUE(sapply(1:length(regparam), FUN= function(x) abs(gradients[[step]][[x]]) < tolerance$gradients))) {
                                print(paste("Gradients convergence,", step, "iterations needed."))
                                break
                        }
                }
        }

        # Compute std error for each estimator
        stderror <- vector(mode = "list", length = length(new_list))
        diagvarcov <- sqrt(diag(solve(mhess)))
        names(stderror) <- names(regparam)
        for (i in 1:length(new_list)) stderror[[i]] <- diagvarcov[i] #ESTO PUEDE SER MAS EFICIENTE

        # Organize results of each iteration
        gradients <- purrr::transpose(gradients)
        parameters <- purrr::transpose(parameters)
        gradientsfinal <- parametersfinal <- namesgradients <- as.numeric()
        for (j in 1:length(new_list)) {
                gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[j]]))
                parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[j]]))
                namesgradients <- cbind(namesgradients, paste0("Gradients ", names(regparam)[j]))
        }
        # Table of results
        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(regparam), namesgradients)
        return(list(results = results.table, final = tail(results.table, 1), standarderror = stderror))
}



# LINK FUNCTION
link <- function(link_function, sum, parameter) {

        if (is.null(link_function)) {
                if (all.vars(ydist)[2] == "Poisson") {
                        sum <- tf$exp(sum)
                        #warning("If Y ~ Poisson, you should use the log link function")
                } else {
                        sum <- sum
                }
        } else if (!is.null(link_function)) {
                if (parameter %in% names(link_function)) {
                        if (link_function[[parameter]] == "log") {
                                sum <- tf$exp(sum)
                        }else if (link_function[[parameter]] == "logit") {
                                sum <- tf$exp(sum) / (1 + tf$exp(sum))
                        }
                } else {
                        sum <- sum
                }
        }
}

