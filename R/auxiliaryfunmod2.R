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

        nbetas <- param <- bnum <- sum <- sumlink <- vector(mode = "list", length = np)
        totalbetas <- sum(as.numeric(unlist(sapply(design_matrix[1:np], ncol))))
        regparam <- multparam <- vector(mode = "list", length = totalbetas)

        nbetas <- lapply(1:np, FUN = function(i) nbetas[[i]] <- sum(as.numeric(unlist(sapply(design_matrix[i], ncol)))))
        bnum <- lapply(1:np, FUN = function(i) bnum[[i]] <- rep(0:(nbetas[[i]]-1)))
        bnumvector <- unlist(bnum, use.names=FALSE)
        names(nbetas) <- names(param) <- names(bnum) <- names(design_matrix)[1:np]
        namesbetas <- unlist(lapply(1:np, FUN = function(i) rep(names(nbetas)[i], nbetas[[i]])))
        initparamvector <- unlist(lapply(1:np, FUN = function(i) rep(initparam[[i]], nbetas[[i]])))

        nameregparam <- unlist(lapply(1:totalbetas, FUN = function(i) paste0("beta", bnumvector[i], namesbetas[i])))

        regparam <- lapply(1:totalbetas, FUN = function(i) assign(nameregparam[i],
                                                                  tf$Variable(initparamvector[i],
                                                                              dtype = tf$float32),
                                                                  envir = .GlobalEnv))
        names(regparam) <- nameregparam

        nbetasvector <- unlist(lapply(1:np, FUN = function(i) rep(1:nbetas[[i]])))

        multparam <- lapply(1:totalbetas, FUN = function(i) tf$multiply(design_matrix[[namesbetas[i]]][, nbetasvector[i]], regparam[[i]]))

        nbetasvector <- unlist(nbetas, use.names = FALSE)

        t <- vector(mode = "list")
        if (np > 1) {
                t <- lapply(1:np, FUN = function(i) t[[i]] <- ifelse(i == 1, 0,
                                                                     Reduce("+", nbetas[[1:(i - 1)]])))
        } else {
                t[[1]] <- 0
        }

        addparam <- function(i) {
                add <- function(x) Reduce("+", x)
                sumparam <- add(multparam[(1 + t[[i]]):(nbetas[[i]] + t[[i]])])
                return(sumparam)
        }

        sum <- lapply(1:np, FUN = function(i) sum[[i]] <- addparam(i))

        sumlink <- lapply(1:np, FUN = function(i) sumlink[[i]] <- link(link_function, sum[[i]], names(nbetas)[i]))

        param <- lapply(1:np, FUN = function(i) param[[i]] <- assign(names(nbetas)[i], sumlink[[i]], envir = .GlobalEnv))
        names(param) <- names(design_matrix)[1:np]


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
        new_list <- lapply(1:totalbetas, FUN = function(i) new_list[[i]] <- regparam[[i]])
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
        maxiter <- 10000

        while(TRUE){
                # Update step
                step <- step + 1

                # Gradient step
                sess$run(train, feed_dict = fd)

                # Parameters and gradients as numeric vectors
                #for (i in 1:length(regparam)) {
                 #       objvariables[[i]] <- as.numeric(sess$run(regparam[[i]]))
                  #      itergrads[[i]] <- as.numeric(sess$run(grads, feed_dict = fd)[[i]])
                #}
                objvariables <- lapply(1:length(regparam), FUN = function(i) objvariables[[i]] <- as.numeric(sess$run(regparam[[i]])))
                itergrads <- lapply(1:length(regparam), FUN = function(i) itergrads[[i]] <- as.numeric(sess$run(grads, feed_dict = fd)[[i]]))


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
        #for(i in 1:length(regparam)) hesslist[[i]] <- tf$gradients(grads[[i]], regparam)
        hesslist <- lapply(1:length(regparam), FUN = function(i) hesslist[[i]] <- tf$gradients(grads[[i]], new_list))
        hess <- tf$stack(values=hesslist, axis=0)
        #hess <- tf$reshape(hess, shape(np, np))
        mhess <- sess$run(hess, feed_dict = fd)
        diagvarcov <- sqrt(diag(solve(mhess)))
        #names(stderror) <- names(var_list)
        #for (i in 1:length(regparam)) stderror[[i]] <- diagvarcov[i] #ESTO PUEDE SER MAS EFICIENTE
        stderror <- lapply(1:length(regparam), FUN = function(i) stderror[[i]] <- diagvarcov[i])
        names(stderror) <- names(regparam)

        # Close tf session
        sess$close()

        # Organize results of each iteration
        gradients <- purrr::transpose(gradients)
        parameters <- purrr::transpose(parameters)
        gradientsfinal <- parametersfinal <- namesgradients <- as.numeric()
        #for (j in 1:length(regparam)) {
         #       gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[j]]))
          #      parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[j]]))
           #     namesgradients <- cbind(namesgradients, paste0("Gradients ", names(regparam)[j]))
        #}
        gradientsfinal <- sapply(1:length(regparam), function(i) gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[i]])))
        parametersfinal <- sapply(1:length(regparam), function(i) parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[i]])))
        namesgradients <- sapply(1:length(regparam), function(i) namesgradients <- cbind(namesgradients, paste0("Gradients ", names(regparam)[i])))

        # Table of results
        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(regparam), namesgradients)
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

        nbetas <- param <- bnum <- sum <- sumlink <- vector(mode = "list", length = np)
        totalbetas <- sum(as.numeric(unlist(sapply(design_matrix[1:np], ncol))))
        regparam <- multparam <- vector(mode = "list", length = totalbetas)

        nbetas <- lapply(1:np, FUN = function(i) nbetas[[i]] <- sum(as.numeric(unlist(sapply(design_matrix[i], ncol)))))
        bnum <- lapply(1:np, FUN = function(i) bnum[[i]] <- rep(0:(nbetas[[i]]-1)))
        bnumvector <- unlist(bnum, use.names=FALSE)
        names(nbetas) <- names(param) <- names(bnum) <- names(design_matrix)[1:np]
        namesbetas <- unlist(lapply(1:np, FUN = function(i) rep(names(nbetas)[i], nbetas[[i]])))
        initparamvector <- unlist(lapply(1:np, FUN = function(i) rep(initparam[[i]], nbetas[[i]])))

        nameregparam <- unlist(lapply(1:totalbetas, FUN = function(i) paste0("beta", bnumvector[i], namesbetas[i])))

        regparam <- lapply(1:totalbetas, FUN = function(i) assign(nameregparam[i],
                                                                  tf$Variable(initparamvector[i],
                                                                              dtype = tf$float32),
                                                                  envir = .GlobalEnv))
        names(regparam) <- nameregparam

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
        #for (i in 1:length(regparam)) new_list[[i]] <- regparam[[i]]
        new_list <- lapply(1:length(regparam), FUN = function(i) new_list[[i]] <- regparam[[i]])

        maxiter <- 10000

        while(TRUE){
                # Update step
                step <- step + 1

                with(tf$GradientTape(persistent = TRUE) %as% tape, {
                        nbetasvector <- unlist(lapply(1:np, FUN = function(i) rep(1:nbetas[[i]])))

                        multparam <- lapply(1:totalbetas, FUN = function(i) tf$multiply(design_matrix[[namesbetas[i]]][, nbetasvector[i]], regparam[[i]]))

                        nbetasvector <- unlist(nbetas, use.names = FALSE)

                        t <- vector(mode = "list")
                        if (np > 1) {
                                t <- lapply(1:np, FUN = function(i) t[[i]] <- ifelse(i == 1, 0,
                                                                                     Reduce("+", nbetas[[1:(i - 1)]])))
                        } else {
                                t[[1]] <- 0
                        }

                        addparam <- function(i) {
                                add <- function(x) Reduce("+", x)
                                sumparam <- add(multparam[(1 + t[[i]]):(nbetas[[i]] + t[[i]])])
                                return(sumparam)
                        }

                        sum <- lapply(1:np, FUN = function(i) sum[[i]] <- addparam(i))

                        sumlink <- lapply(1:np, FUN = function(i) sumlink[[i]] <- link(link_function, sum[[i]], names(nbetas)[i]))

                        param <- lapply(1:np, FUN = function(i) param[[i]] <- assign(names(nbetas)[i], sumlink[[i]], envir = .GlobalEnv))
                        names(param) <- names(design_matrix)[1:np]


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
                        #for(i in 1:length(new_list)) hesslist[[i]] <- tape$gradient(grads[[i]], new_list)
                        hesslist <- lapply(1:length(new_list), FUN = function(i) hesslist[[i]] <- tape$gradient(grads[[i]], new_list))
                        mhess <- as.matrix(tf$stack(values=hesslist, axis=0))
                })


                # Compute gradientes
                seloptimizer$apply_gradients(purrr::transpose(list(grads, new_list)))

                # Save loss value
                loss[[step]] <- as.numeric(loss_value)

                # Save gradients values
                gradients[[step]] <- grads

                # Parameters and gradients as numeric vectors
                #for (i in 1:length(regparam)) {
                #       objvariables[[i]] <- as.numeric(get(names(regparam)[i]))
                #      gradients[[step]][[i]] <- as.numeric(gradients[[step]][[i]])
                #}
                objvariables <- lapply(1:length(regparam), FUN = function(i) objvariables[[i]] <- as.numeric(get(names(regparam)[i])))
                gradients[[step]] <- lapply(1:length(regparam),  FUN = function(i)gradients[[step]][[i]] <- as.numeric(gradients[[step]][[i]]))

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
        #for (i in 1:length(new_list)) stderror[[i]] <- diagvarcov[i] #ESTO PUEDE SER MAS EFICIENTE
        stderror <- lapply(1:length(new_list), FUN = function(i) stderror[[i]] <- diagvarcov[i])

        # Organize results of each iteration
        gradients <- purrr::transpose(gradients)
        parameters <- purrr::transpose(parameters)
        gradientsfinal <- parametersfinal <- namesgradients <- as.numeric()
        #for (j in 1:length(new_list)) {
        #       gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[j]]))
        #      parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[j]]))
        #     namesgradients <- cbind(namesgradients, paste0("Gradients ", names(regparam)[j]))
        #}
        gradientsfinal <- sapply(1:length(new_list), function(i) gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[i]])))
        parametersfinal <- sapply(1:length(new_list), function(i) parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[i]])))
        namesgradients <- sapply(1:length(new_list), function(i) namesgradients <- cbind(namesgradients, paste0("Gradients ", names(regparam)[i])))

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
        return(sum)
}


comparisonreg <- function(ydist, formulas, data, link_function, fixparam, initparam, lower, upper, method) {
        distributionsr <- list(Bernoulli = "dbinom", Beta = "dbeta", Exponential = "dexp", Gamma = "dgamma",
                               Normal = "dnorm", Uniform = "dunif", Poisson = "dpois", FWE = "dFWE")

        parametersr <- list(loc = "mean", scale = "sd", concentration1 = "shape1", concentration2 = "shape2",
                            concentration = "shape", low = "min", high = "max", lambda = "lambda", mu = "mu",
                            sigma = "sigma")


        #if (!is.null(fixparam)) for (i in 1:length(fixparam)) names(fixparam)[i] <- parametersr[[match(names(fixparam)[i], names(parametersr))]]
        #if (!is.null(initparam)) for (i in 1:length(initparam)) names(initparam)[i] <- parametersr[[match(names(initparam)[i], names(parametersr))]]
        #if (!is.null(link_function)) for (i in 1:length(link_function)) names(link_function)[i] <- parametersr[[match(names(link_function)[i], names(parametersr))]]

        if (!is.null(fixparam)) names(fixparam) <- lapply(1:length(fixparam), FUN = function(i) names(fixparam)[i] <- parametersr[[match(names(fixparam)[i], names(parametersr))]])
        if (!is.null(initparam)) names(initparam) <- lapply(1:length(initparam), FUN = function(i) names(initparam)[i] <- parametersr[[match(names(initparam)[i], names(parametersr))]])
        if (!is.null(link_function)) names(link_function) <- lapply(1:length(link_function), FUN = function(i) names(link_function)[i] <- parametersr[[match(names(link_function)[i], names(parametersr))]])

        #JAIME
        link = list(over=c("mu", "sigma"), fun = c("log_link", "logit_link"))

        link <- vector(mode = "list", length = 2 * length(link_function))
        if (!is.null(link_function)) {
                p <- 0
                for (i in 1:length(link_function)) {
                        link[[i + p]] <- names(link_function)[i]
                        link[[i + p + 1]] <- paste0(link_function[[i]], "_link")
                        names(link)[[i + p]] <- "over"
                        names(link)[[i + p + 1]] <- "fun"
                        p <- 1
                }
        }

        estimation <- maxlogLreg(formulas = formulas, y_dist = ydist, data = data,
                               link = link, lower = lower, upper = upper, start = initparam,
                               optimizer = method)
        return(estimation)


}
