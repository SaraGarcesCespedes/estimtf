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
disableagerreg <- function(data, dist, design_matrix, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np) {

        # Disable eager execution
        tf$compat$v1$disable_eager_execution()


        # Create list to store the parameters to be estimated
        #var_list <- place_list <- vector(mode = "list", length = np)
        #names(var_list) <- names(initparam)
        #names(place_list) <- paste0("X", names(initparam))

        # Create placeholders
        Y <- tf$compat$v1$placeholder(dtype = tf$float64, name = "y_data")
        y_data <- design_matrix$y

        #x_data <- design_matrix$data_reg[, -1]

        nvar <- betas <- X <- param <- vector(mode = "list", length = np) #NO CREO QUE SEA NECESARIO CREAR LISTA BETAS
        names(nvar) <- names(betas) <- names(X) <- names(param) <- names(initparam)
        for (i in 1:np) {
                X[[i]] <- eval(parse(text = paste("design_matrix$", names(initparam)[i], sep = "")))
                nvar[[i]] <- dim(X[[i]])[2]
                betas[[i]] <- assign(paste0("betas", names(initparam)[i]), tf$Variable(tf$zeros(list(nvar[[i]], 1L), dtype = tf$float64), name = paste0("betas", names(initparam)[i])))
                param[[i]] <- assign(names(initparam[i]), tf$matmul(X[[i]], betas[[i]]))
        }



        #for (i in 1:np) {
         #       X <- eval(parse(text = paste("design_matrix$", names(initparam)[i], sep = "")))
          #      nvar <- dim(X)[2]
           #     betas <- assign(paste0("betas", names(initparam)[i]), tf$Variable(tf$zeros(list(nvar, 1L), dtype = tf$float64), name = paste0("betas", names(initparam)[i])))
            #    param <- assign(paste0(names(initparam)[i]), tf$matmul(X, betas))
        #}


        # Create a list with all parameters, fixed and not fixed
        vartotal <- append(fixparam, param)

        # Create vectors to store parameters, gradientes and loss values of each iteration
        loss <- new_list <- parameters <- gradients <- itergrads <- objvariables <- vector(mode = "list")

        # Create list with variables without names
        for (i in 1:np) new_list[[i]] <- betas[[i]]

        # Define loss function depending on the distribution
        if (dist == "Poisson") {
                loss_value <- tf$reduce_sum(-Y * (tf$math$log(vartotal[["lambda"]])) + vartotal[["lambda"]])
        } else if (dist == "FWE") {
                loss_value <- -tf$reduce_sum(tf$math$log(vartotal[["mu"]] + vartotal[["sigma"]] / (Y ^ 2))) - tf$reduce_sum(vartotal[["mu"]] * Y - vartotal[["sigma"]] / Y) + tf$reduce_sum(tf$math$exp(vartotal[["mu"]] * Y - vartotal[["sigma"]] / Y))
        } else if (dist == "Instantaneous Failures") {
                loss_value <- -tf$reduce_sum(tf$math$log((((vartotal[["lambda"]] ^ 2) + Y - 2 * vartotal[["lambda"]]) * tf$math$exp(-Y / vartotal[["lambda"]])) / ((vartotal[["lambda"]] ^ 2) * (vartotal[["lambda"]] - 1))))
        } else {
                #density <- do.call(what = dist, vartotal)
                #loss_value <- tf$negative(tf$reduce_sum(density$log_prob(value = Y)))
                n <- length(y_data)
                loss_value <- (n / 2) * tf$math$log(vartotal[["scale"]]^2) + (1 / (2 * vartotal[["scale"]]^2)) * tf$reduce_sum((Y - vartotal[["loc"]])^2)
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
        fd <- dict(Y = y_data)

        # Initialize step
        step <- 0

        while(TRUE){
                # Update step
                step <- step + 1

                # Gradient step
                sess$run(train, feed_dict = fd)

                # Parameters and gradients as numeric vectors
                for (i in 1:length(new_list)) {
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
                        } else if (isTRUE(sapply(1:length(new_list), FUN= function(x) abs(parameters[[step]][[x]]) < tolerance$parameters))) {
                                print(paste("Parameters convergence,", step, "iterations needed."))
                                break
                        } else if (isTRUE(sapply(1:length(new_list), FUN= function(x) abs(gradients[[step]][[x]]) < tolerance$gradients))) {
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
eagerreg <- function(data, dist, design_matrix, fixparam, initparam, opt, hyperparameters, maxiter, tolerance, np) {

        # Create list to store the parameters to be estimated
        #var_list <- vector(mode = "list", length = np)
        #names(var_list) <- names(initparam)

        # Create tf Variables
        #for (i in 1:np) {
         #       var_list[[i]] <- assign(names(initparam)[i], tf$Variable(initparam[[i]], dtype = tf$float32, name = names(initparam)[i]))
        #}

        y_data <- design_matrix$y

        #x_data <- design_matrix$data_reg[, -1]

        nvar <- betas <- X <- param <- intercept <- vector(mode = "list", length = np) #NO CREO QUE SEA NECESARIO CREAR LISTA BETAS
        names(nvar) <- names(betas) <- names(X) <- names(param) <-  names(intercept) <-  names(initparam)
        #for (i in 1:np) {
         #       X[[i]] <- eval(parse(text = paste("design_matrix$", names(initparam)[i], sep = "")))
          #      nvar[[i]] <- ncol(X[[i]])
           #     betas[[i]] <- assign(paste0("betas", names(initparam)[i]), tf$Variable(tf$ones(list(nvar[[i]], 1L), dtype = tf$float64), name = paste0("betas", names(initparam)[i])))
            #    param[[i]] <- assign(names(initparam[i]), tf$matmul(X[[i]], betas[[i]]))
        #}

        for (i in 1:np) {
                nvar[[i]] <- eval(parse(text = paste("ncol(design_matrix$", names(initparam)[i], ")-1", sep = "")))
                X[[i]] <- eval(parse(text = paste("design_matrix$", names(initparam)[i], "[, -1]", sep = "")))
                betas[[i]] <- assign(paste0("betas", names(initparam)[i]), tf$Variable(tf$ones(list(nvar[[i]], 1L), dtype = tf$float64), name = paste0("betas", names(initparam)[i]), trainable =TRUE))
                intercept[[i]] <- assign(paste0("intercept", names(initparam)[i]), tf$Variable(1.0, dtype = tf$float64, name = paste0("intercept", names(initparam)[i]), trainable =TRUE))
                #matmul no funciona si es una sola variable
                #param[[i]] <- assign(names(initparam[i]), tf$matmul(tf$constant(X[[i]], dtype = tf$float64), betas[[i]]) + intercept[[i]])
                param[[i]] <- assign(names(initparam[i]), tf$exp(tf$constant(X[[i]], dtype = tf$float64) * betas[[i]] + intercept[[i]]))

        }

        X <- eval(parse(text = paste("design_matrix$", names(initparam)[1], "[, -1]", sep = "")))

        beta <- tf$Variable(1.0, dtype = tf$float32)
        intercept <- tf$Variable(1.0, dtype = tf$float32)
        lambda <- tf$exp(x * beta + intercept)

        vartotal <- append(fixparam, lambda)
        names(vartotal) <- "lambda"

        if (!is.null(fixparam)) {
        for (j in length(fixparam)) {
                fixparam[[j]] <- tf$constant(fixparam[[j]], dtype = tf$float64)
        }
        }
        # Create a list with all parameters, fixed and not fixed
        vartotal <- append(fixparam, param)
        paramtotal <- append(betas, intercept)
        # Define optimizer
        seloptimizer <- do.call(what = opt, hyperparameters)

        # Initialize step
        step <- 0

        # Create vectors to store parameters, gradientes and loss values of each iteration
        loss <- new_list <- parameters <- gradients <- hesslist <- objvariables <- vector(mode = "list")

        # Create list with variables without names
        #for (i in 1:np) new_list[[i]] <- betas[[i]]
        for (i in 1:length(vartotal)) new_list[[i]] <- vartotal[[i]]

        while(TRUE){
                # Update step
                step <- step + 1

                with(tf$GradientTape(persistent = TRUE) %as% tape, {
                        # Define loss function depending on the distribution
                        if (dist == "Poisson") {
                                #loss_value <- tf$reduce_sum(-y_data * (tf$math$log(vartotal[["lambda"]])) + vartotal[["lambda"]])
                                loss_value <- tf$reduce_sum(-y_data * (tf$math$log(lambda)) + lambda)
                        } else if (dist == "FWE") {
                                loss_value <- -tf$reduce_sum(tf$math$log(vartotal[["mu"]] + vartotal[["sigma"]] / (y_data ^ 2))) - tf$reduce_sum(vartotal[["mu"]] * y_data - vartotal[["sigma"]] / y_data) + tf$reduce_sum(tf$math$exp(vartotal[["mu"]] * y_data - vartotal[["sigma"]] / y_data))
                        } else if (dist == "Instantaneous Failures") {
                                loss_value <- -tf$reduce_sum(tf$math$log((((vartotal[["lambda"]] ^ 2) + y_data - 2 * vartotal[["lambda"]]) * tf$math$exp(-y_data / vartotal[["lambda"]])) / ((vartotal[["lambda"]] ^ 2) * (vartotal[["lambda"]] - 1))))
                        } else {
                                density <- do.call(what = dist, vartotal)
                                loss_value <- tf$negative(tf$reduce_sum(density$log_prob(value = y_data)))
                        }
                        #grads <- tape$gradient(loss_value, list(beta, intercept))
                        # Compute Hessian matrixin
                        #for(i in 1:length(paramtotal)) hesslist[[i]] <- tape$gradient(grads[[i]], c(betaslambda, interceptlambda))
                        #mhess <- as.matrix(tf$stack(values=hesslist, axis=0))
                })
                grads <- tape$gradient(loss_value, list(beta, intercept))


                # Compute gradientes
                seloptimizer$apply_gradients(purrr::transpose(list(grads, c(beta, intercept))))

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

