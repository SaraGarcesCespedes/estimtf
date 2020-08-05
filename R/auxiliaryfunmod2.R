#------------------------------------------------------------------------
# Estimation of regression parameters (disable eager execution) ---------
#------------------------------------------------------------------------
disableagerreg <- function(data, dist, design_matrix, fixparam, initparam, argumdist, opt, hyperparameters, maxiter, tolerance, np, link_function, ydist, distnotf) {

        # Disable eager execution
        tf$compat$v1$disable_eager_execution()

        # Create placeholders
        Y <- tf$compat$v1$placeholder(dtype = tf$float32, name = "y_data")
        y_data <- design_matrix$y

        nbetas <- initparamvector <- param <- sum <- sumlink <- namesparam <- vector(mode = "list", length = np)
        totalbetas <- sum(as.numeric(unlist(sapply(design_matrix[1:np], ncol))))
        regparam <- multparam <- vector(mode = "list", length = totalbetas)

        nbetas <- lapply(1:np, FUN = function(i) nbetas[[i]] <- sum(as.numeric(unlist(sapply(design_matrix[i], ncol)))))
        namesparam <- lapply(1:np, FUN = function(i) namesparam[[i]] <- lapply(design_matrix[i], colnames))

        namesparamvector <- unlist(namesparam, use.names=FALSE)
        namesparamvectornew <- numeric()
        namesparamvectornew <- sapply(1:length(namesparamvector),
                                      FUN = function(i) namesparamvectornew[i] <- ifelse(namesparamvector[i] == "(Intercept)",
                                                                                         gsub("[()]","", namesparamvector[i]),
                                                                                         namesparamvector[i]))

        names(nbetas) <- names(param) <- names(design_matrix)[1:np]
        namesbetas <- unlist(lapply(1:np, FUN = function(i) rep(names(nbetas)[i], nbetas[[i]])))

        t <- vector(mode = "list")
        if (np > 1) {
                t <- lapply(1:np,
                            FUN = function(i) t[[i]] <- ifelse(i == 1, 0,
                                                               Reduce("+", nbetas[[1:(i - 1)]])))
        } else {
                t[[1]] <- 0
        }

        initvalues <- function(i) {
                if (is.numeric(initparam[[i]])) {
                        vector <- rep(initparam[[i]], nbetas[[i]])
                } else {
                        selparam <- namesparamvectornew[(1 + t[[i]]):(t[[i]] + nbetas[[i]])]
                        paramprovided <- match(names(initparam[[i]]), selparam)
                        namesparamprovided <- names(initparam[[i]])
                        missingparam <- selparam[-paramprovided]
                        initparam[[i]] <- append(initparam[[i]], rep(1.0, length(missingparam)))
                        names(initparam[[i]]) <- c(namesparamprovided, missingparam)
                        # order of initparam and namesparamvectornew must be the same
                        initparam[[i]] <- initparam[[i]][selparam]
                        vector <- unlist(initparam[[i]], use.names=FALSE)
                }
        }

        initparamvector <- lapply(1:np, FUN = function(i) initparamvector[[i]] <- initvalues(i))
        initparamvector <- unlist(initparamvector, use.names=FALSE)

        nameregparam <- unlist(lapply(1:totalbetas, FUN = function(i) paste0(namesparamvectornew[i],
                                                                             "_", namesbetas[i])))

        regparam <- lapply(1:totalbetas, FUN = function(i) assign(nameregparam[i],
                                                                  tf$Variable(initparamvector[i],
                                                                              dtype = tf$float32),
                                                                  envir = .GlobalEnv))
        names(regparam) <- nameregparam

        nbetasvector <- unlist(lapply(1:np, FUN = function(i) rep(1:nbetas[[i]])))

        multparam <- lapply(1:totalbetas, FUN = function(i) tf$multiply(design_matrix[[namesbetas[i]]][, nbetasvector[i]], regparam[[i]]))

        #nbetasvector <- unlist(nbetas, use.names = FALSE)

        #es posible que me genere problemas
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

        n <- length(y_data)
        X <- Y
        # Define loss function depending on the distribution
        if (all.vars(ydist)[2] %in% distnotf) {
                loss_value <- lossfun(dist, vartotal, X)
        } else {
                density <- do.call(what = dist, vartotal)
                loss_value <- tf$negative(tf$reduce_sum(density$log_prob(value = X)))
        }

        # Compute gradients
        new_list <- lapply(1:length(regparam), FUN = function(i) new_list[[i]] <- regparam[[i]])
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
        hesslist <- lapply(1:length(regparam), FUN = function(i) hesslist[[i]] <- tf$gradients(grads[[i]], new_list))
        hess <- tf$stack(values=hesslist, axis=0)
        mhess <- sess$run(hess, feed_dict = fd)
        diagvarcov <- sqrt(diag(solve(mhess)))
        stderror <- lapply(1:length(regparam), FUN = function(i) stderror[[i]] <- diagvarcov[i])
        names(stderror) <- names(regparam)

        # Close tf session
        sess$close()

        # Organize results of each iteration
        gradients <- purrr::transpose(gradients)
        parameters <- purrr::transpose(parameters)
        gradientsfinal <- parametersfinal <- namesgradients <- as.numeric()
        gradientsfinal <- sapply(1:length(regparam), function(i) gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[i]])))
        parametersfinal <- sapply(1:length(regparam), function(i) parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[i]])))
        namesgradients <- sapply(1:length(regparam), function(i) namesgradients <- cbind(namesgradients, paste0("Gradients ", names(regparam)[i])))

        # Table of results
        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(regparam), namesgradients)
        return(list(results = results.table, final = tail(results.table, 1), standarderror = stderror))


}

#------------------------------------------------------------------------
# Estimation of regression parameters (with eager execution) ------------
#------------------------------------------------------------------------
eagerreg <- function(data, dist, design_matrix, fixparam, initparam, argumdist, opt, hyperparameters, maxiter, tolerance, np, link_function, ydist, distnotf) {

        y_data <- as.double(design_matrix$y)

        nbetas <- initparamvector <- param <- sum <- sumlink <- namesparam <- vector(mode = "list", length = np)
        totalbetas <- sum(as.numeric(unlist(sapply(design_matrix[1:np], ncol))))
        regparam <- multparam <- vector(mode = "list", length = totalbetas)

        nbetas <- lapply(1:np, FUN = function(i) nbetas[[i]] <- sum(as.numeric(unlist(sapply(design_matrix[i], ncol)))))
        namesparam <- lapply(1:np, FUN = function(i) namesparam[[i]] <- lapply(design_matrix[i], colnames))

        namesparamvector <- unlist(namesparam, use.names=FALSE)
        namesparamvectornew <- numeric()
        namesparamvectornew <- sapply(1:length(namesparamvector),
                                      FUN = function(i) namesparamvectornew[i] <- ifelse(namesparamvector[i] == "(Intercept)",
                                                                                         gsub("[()]","", namesparamvector[i]),
                                                                                         namesparamvector[i]))

        names(nbetas) <- names(param) <- names(design_matrix)[1:np]
        namesbetas <- unlist(lapply(1:np, FUN = function(i) rep(names(nbetas)[i], nbetas[[i]])))

        t <- vector(mode = "list")
        if (np > 1) {
                t <- lapply(1:np,
                            FUN = function(i) t[[i]] <- ifelse(i == 1, 0,
                                                               Reduce("+", nbetas[[1:(i - 1)]])))
        } else {
                t[[1]] <- 0
        }

        initvalues <- function(i) {
                if (is.numeric(initparam[[i]])) {
                        vector <- rep(initparam[[i]], nbetas[[i]])
                } else {
                        selparam <- namesparamvectornew[(1 + t[[i]]):(t[[i]] + nbetas[[i]])]
                        paramprovided <- match(names(initparam[[i]]), selparam)
                        namesparamprovided <- names(initparam[[i]])
                        missingparam <- selparam[-paramprovided]
                        initparam[[i]] <- append(initparam[[i]], rep(1.0, length(missingparam)))
                        names(initparam[[i]]) <- c(namesparamprovided, missingparam)
                        # order of initparam and namesparamvectornew must be the same
                        initparam[[i]] <- initparam[[i]][selparam]
                        vector <- unlist(initparam[[i]], use.names=FALSE)
                }
        }

        initparamvector <- lapply(1:np, FUN = function(i) initparamvector[[i]] <- initvalues(i))
        initparamvector <- unlist(initparamvector, use.names=FALSE)

        nameregparam <- unlist(lapply(1:totalbetas, FUN = function(i) paste0(namesparamvectornew[i],
                                                                             "_", namesbetas[i])))

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

        # Define optimizer
        seloptimizer <- do.call(what = opt, hyperparameters)

        # Initialize step
        step <- 0

        # Create vectors to store parameters, gradientes and loss values of each iteration
        loss <- new_list <- parameters <- gradients <- hesslist <- objvariables <- vector(mode = "list")
        new_list <- lapply(1:length(regparam), FUN = function(i) new_list[[i]] <- regparam[[i]])

        while(TRUE){
                # Update step
                step <- step + 1

                with(tf$GradientTape(persistent = TRUE) %as% tape, {
                        nbetasvector <- unlist(lapply(1:np, FUN = function(i) rep(1:nbetas[[i]])))

                        multparam <- lapply(1:totalbetas, FUN = function(i) tf$multiply(design_matrix[[namesbetas[i]]][, nbetasvector[i]], regparam[[i]]))

                        #nbetasvector <- unlist(nbetas, use.names = FALSE)

                        #es posible que me genere problemas
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

                        n <- length(y_data)
                        X <- y_data
                        # Define loss function depending on the distribution
                        if (all.vars(ydist)[2] %in% distnotf) {
                                loss_value <- lossfun(dist, vartotal, X)
                        } else {
                                density <- do.call(what = dist, vartotal)
                                loss_value <- tf$negative(tf$reduce_sum(density$log_prob(value = X)))
                        }
                        grads <- tape$gradient(loss_value, new_list)
                        # Compute Hessian matrixin
                        hesslist <- lapply(1:length(new_list), FUN = function(i) hesslist[[i]] <- tape$gradient(grads[[i]], new_list))
                        mhess <- as.matrix(tf$stack(values=hesslist, axis=0))
                })


                # Compute gradientes
                seloptimizer$apply_gradients(purrr::transpose(list(grads, new_list)))

                # Save loss value
                loss[[step]] <- as.numeric(loss_value)

                # Save gradients values
                gradients[[step]] <- grads

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
        stderror <- lapply(1:length(new_list), FUN = function(i) stderror[[i]] <- diagvarcov[i])

        # Organize results of each iteration
        gradients <- purrr::transpose(gradients)
        parameters <- purrr::transpose(parameters)
        gradientsfinal <- parametersfinal <- namesgradients <- as.numeric()
        gradientsfinal <- sapply(1:length(new_list), function(i) gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[i]])))
        parametersfinal <- sapply(1:length(new_list), function(i) parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[i]])))
        namesgradients <- sapply(1:length(new_list), function(i) namesgradients <- cbind(namesgradients, paste0("Gradients ", names(regparam)[i])))

        # Table of results
        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(regparam), namesgradients)
        return(list(results = results.table, final = tail(results.table, 1), standarderror = stderror))
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

#------------------------------------------------------------------------
# Link function ---------------------------------------------------------
#------------------------------------------------------------------------
link <- function(link_function, sum, parameter) {

        if (is.null(link_function)) {
                if (all.vars(ydist)[2] == "Poisson") {
                        sum <- tf$exp(sum)
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

