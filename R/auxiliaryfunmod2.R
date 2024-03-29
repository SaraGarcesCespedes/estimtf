#------------------------------------------------------------------------
# Estimation of regression parameters (disable eager execution) ---------
#------------------------------------------------------------------------
disableagerreg <- function(data, dist, design_matrix, fixparam, initparam, argumdist, opt, hyperparameters, maxiter, tolerance, np, link_function, ydist, distnotf, optimizer) {

        # Disable eager execution
        tensorflow::tf$compat$v1$disable_eager_execution()

        Y <- tensorflow::tf$compat$v1$placeholder(dtype = tf$float32, name = "y_data")
        y_data <- design_matrix$y
        n <- length(y_data)


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
                            FUN = function(i) t[[i]] <- ifelse(i == 1, 0, Reduce("+", nbetas[[1:(i - 1)]])))
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
                                                                  tensorflow::tf$Variable(initparamvector[i],
                                                                                          dtype = tf$float32),
                                                                  envir = .GlobalEnv))
        names(regparam) <- nameregparam

        nbetasvector <- unlist(lapply(1:np, FUN = function(i) rep(1:nbetas[[i]])))

        multparam <- lapply(1:totalbetas, FUN = function(i) tensorflow::tf$multiply(design_matrix[[namesbetas[i]]][, nbetasvector[i]], regparam[[i]]))

        #nbetasvector <- unlist(nbetas, use.names = FALSE)

        #es posible que me genere problemas
        addparam <- function(i) {
                add <- function(x) Reduce("+", x)
                sumparam <- add(multparam[(1 + t[[i]]):(nbetas[[i]] + t[[i]])])
                return(sumparam)
        }


        sum <- lapply(1:np, FUN = function(i) sum[[i]] <- addparam(i))


        sumlink <- lapply(1:np, FUN = function(i) sumlink[[i]] <- link(link_function, sum[[i]], names(nbetas)[i], ydist))

        param <- lapply(1:np, FUN = function(i) param[[i]] <- assign(names(nbetas)[i], sumlink[[i]], envir = .GlobalEnv))
        names(param) <- names(design_matrix)[1:np]

        # Create a list with all parameters, fixed and not fixed
        vartotal <- append(fixparam, param)

        # Create vectors to store parameters, gradientes and loss values of each iteration
        loss <- new_list <- parameters <- gradients <- itergrads <- objvariables <- vector(mode = "list")

        X <- Y

        # Define loss function depending on the distribution
        if (all.vars(ydist)[2] %in% distnotf) {
                loss_value <- lossfun_mlereg(dist, vartotal, X, n)
        } else {
                density <- do.call(what = dist, vartotal)
                loss_value <- tensorflow::tf$negative(tensorflow::tf$reduce_sum(density$log_prob(value = X)))
        }

        # Compute gradients
        new_list <- lapply(1:length(regparam), FUN = function(i) new_list[[i]] <- regparam[[i]])
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

                if (is.na(loss[[step]])){
                        stop(paste0("The process failed because the loss value in the last iterarion is NaN \n",
                                    "Follow these recommendations and start the process again: \n",
                                    "1. Reduce the learning rate. \n",
                                    "2. Check your input data as it is possible that some of the values are neither \n",
                                    "integer nor float. \n",
                                    "3. Change the initial values provided for the parameters. \n",
                                    "4. Try different optimizers. \n",
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

        # TODAVIA FALTAAAAA ARREGLAR ESTO
        # Compute Hessian matrix
        hesslist <- stderror <- vector(mode = "list", length = length(regparam))
        hesslist <- lapply(1:length(regparam), FUN = function(i) hesslist[[i]] <- tensorflow::tf$gradients(grads[[i]], new_list))
        hess <- tensorflow::tf$stack(values=hesslist, axis=0)
        mhess <- sess$run(hess, feed_dict = fd)
        varcov <- hessian_matrix_try(mhess)

        if (!is.null(varcov)) {
                std_error_vector <- sqrt(diag(varcov))
                stderror <- lapply(1:length(regparam), FUN = function(i) stderror[[i]] <- std_error_vector[i])
                names(stderror) <- names(regparam)
        } else {
                stderror <- NULL
        }


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
        names_param <- names(design_matrix)[1:np]
        names_new <- vector(mode = "numeric", length = length(names_param))
        names_new <- sapply(1:length(names_param), FUN = function(i) names_new[i] <- parameter_name_R(names_param[i], all.vars(ydist)[2]))

        results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
        colnames(results.table) <- c("loss", names(regparam), namesgradients)
        outputs <- list(nbetas = nbetas, ntotalbetas = length(totalbetas), n = n,
                        type = "MLEregtf", np = np, names = namesparamvector,
                        estimates = tail(results.table[, 2:(totalbetas + 1)], 1),
                        convergence = convergence, names_regparam = names_new)
        result <- list(results = results.table, vcov = varcov, standarderror = stderror,
                       outputs = outputs)
        return(result)


}


#------------------------------------------------------------------------
# Loss function for distributions not included in TF --------------------
#------------------------------------------------------------------------
lossfun_mlereg <- function(dist, vartotal, X, n) {
        if (dist == "FWE") {
                loss <- -tensorflow::tf$reduce_sum(tensorflow::tf$math$log(vartotal[["mu"]] + vartotal[["sigma"]] / (X ^ 2))) -
                        tensorflow::tf$reduce_sum(vartotal[["mu"]] * X - vartotal[["sigma"]] / X) +
                        tensorflow::tf$reduce_sum(tf$math$exp(vartotal[["mu"]] * X - vartotal[["sigma"]] / X))
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
                        (1 / vartotal[["scale"]]) * tensorflow::tf$reduce_sum(tf$abs(X - vartotal[["loc"]]))
        } else if (dist == "Logistic") {

                logits <- vartotal[["logits"]]
                logits <- tf$reshape(logits, shape(n, 1))
                entropy <- tf$nn$sigmoid_cross_entropy_with_logits(labels = X, logits = logits)
                loss <- tf$reduce_mean(entropy)
                #loss <- tensorflow::tf$reduce_sum(-X * vartotal[["logits"]] + tensorflow::tf$math$log(1 + tensorflow::tf$exp(vartotal[["logits"]])))
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
# Link function ---------------------------------------------------------
#------------------------------------------------------------------------
link <- function(link_function, sum, parameter, ydist) {

        if (is.null(link_function)) {

                sum <- sum

        } else if (!is.null(link_function)) {
                if (parameter %in% names(link_function)) {
                        if (link_function[[parameter]] == "log") {
                                sum <- tensorflow::tf$exp(sum)
                        }else if (link_function[[parameter]] == "logit") {
                                sum <- tensorflow::tf$exp(sum) / (1 + tensorflow::tf$exp(sum))
                        }else if (link_function[[parameter]] == "squared") {
                                sum <- sum ^ 2
                        }else if (link_function[[parameter]] == "identity") {
                                sum <- sum
                        }
                } else {
                        sum <- sum
                }
        }
        return(sum)
}

#------------------------------------------------------------------------
# Hessian Matrix Error --------------------------------------------------
#------------------------------------------------------------------------

hessian_matrix_try <- function(mhess){
        tryCatch(
                expr = {
                        diagvarcov <- solve(mhess)
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
