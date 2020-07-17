y_data <- y

NUM_EXAMPLES <- 1000
x <- tf$random$normal(shape = shape(NUM_EXAMPLES))

beta <- tf$Variable(1.0, dtype = tf$float32)
intercept <- tf$Variable(1.0, dtype = tf$float32)
lambda <- tf$exp(x * beta + intercept)

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




# DISABLES EAGER

y_data <- design_matrix$y
beta <- tf$Variable(1.0, dtype = tf$float64)
intercept <- tf$Variable(1.0, dtype = tf$float64)
x <- tf$constant(design_matrix$data_reg[,2], dtype = tf$float64)
lambda <- tf$exp(x * beta + intercept)

# Define loss function depending on the distribution
if (dist == "Poisson") {
        #loss_value <- tf$reduce_sum(-Y * (tf$math$log(vartotal[["lambda"]])) + vartotal[["lambda"]])
        loss_value <- tf$reduce_sum(-Y * (tf$math$log(lambda) +lambda))

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
grads <- tf$gradients(loss_value, list(beta, intercept))

seloptimizer <- do.call(what = opt, hyperparameters)
train <- eval(parse(text = "seloptimizer$minimize(loss_value)"))

# Initialize the variables and open the session
init <- tf$compat$v1$initialize_all_variables()
sess <- tf$compat$v1$Session()
sess$run(init)

# Create dictionary to feed data into graph
fd <- dict(Y = y_data)
