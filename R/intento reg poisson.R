tf$compat$v1$disable_eager_execution()

Y <- tf$compat$v1$placeholder(dtype = tf$float64, name = "y_data")
X <- tf$compat$v1$placeholder(dtype = tf$float64, name = "x_data")
y_data <- design_matrix$y
#x_data <- design_matrix$lambda[, -1]
x_data <- design_matrix$lambda

#beta <- tf$Variable(0.0, dtype = tf$float64)
beta <- tf$Variable(tf$zeros(list(2, 1L), dtype = tf$float64))
#intercept <- tf$Variable(0.0, dtype = tf$float64)
lambda1 <- tf$exp(tf$matmul(X, beta))

if (dist == "Poisson") {
        loss_value <- tf$reduce_sum(-Y * (tf$math$log(lambda1)) + lambda1)
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


grads <- tf$gradients(loss_value, list(beta))

# Define optimizer
seloptimizer <- do.call(what = opt, hyperparameters)
train <- eval(parse(text = "seloptimizer$minimize(loss_value)"))

# Initialize the variables and open the session
init <- tf$compat$v1$initialize_all_variables()
sess <- tf$compat$v1$Session()
sess$run(init)

# Create dictionary to feed data into graph
fd <- dict(Y = y_data, X = x_data)

# Initialize step
step <- 0
loss <- new_list <- parameters <- gradients <- itergrads <- objvariables <- vector(mode = "list")
param <- list(beta)
maxiter <- 10000
while(TRUE){
        # Update step
        step <- step + 1

        # Gradient step
        sess$run(train, feed_dict = fd)

        # Parameters and gradients as numeric vectors
        np <- 1
        for (i in 1:np) {
                objvariables[[i]] <- as.numeric(sess$run(param[[i]]))
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
                } else if (isTRUE(sapply(1:length(param), FUN= function(x) abs(parameters[[step]][[x]]) < tolerance$parameters))) {
                        print(paste("Parameters convergence,", step, "iterations needed."))
                        break
                } else if (isTRUE(sapply(1:length(param), FUN= function(x) abs(gradients[[step]][[x]]) < tolerance$gradients))) {
                        print(paste("Gradients convergence,", step, "iterations needed."))
                        break
                }
        }
}

# Compute Hessian matrix
hesslist <- stderror <- vector(mode = "list", length = np)
for(i in 1:np) hesslist[[i]] <- tf$gradients(grads[[i]], param)
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
var_list <- vector(mode = "list", length = 2)
names(var_list) <- c("beta", "intercept")
# Table of results
results.table <- cbind(as.numeric(loss), parametersfinal, gradientsfinal)
colnames(results.table) <- c("loss", names(var_list), namesgradients)
return(list(results = results.table, final = tail(results.table, 1), standarderror = stderror))

