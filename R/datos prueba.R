formulas <- list(loc.fo = ~ x + x1, scale.fo = ~ x)
n <- 1000
x <- runif(n = n, 0, 6)
x1 <- runif(n = n, 0, 6)
y <- rnorm(n = n, mean = -2 + 3 * x + 9* x1, sd = 3 + 3* x)
data <- data.frame(y = y, x = x, x1=x1)

formulas <- list(lambda.fo = ~ x)
n <- 1000
x <- runif(n = n, -1, 1)
y <- rpois(n = n, lambda = exp(-2 + 3 * x))
data <- data.frame(y = y, x = x)



true_mu    <- 0
true_sigma <- 0
x <- rFWE(n=1000, mu=true_mu, sigma=true_sigma)


size = 1000
b0 <- -2
b1 <- 0.9
g0 <- 2
g1 <- -6.7
x1 <- runif(n=size)
x2 <- runif(n=size)
mu <- exp(b0 + b1 * x1)
sig <- exp(g0 + g1 * x2)
y <- rFWE(n=size, mu=mu, sigma=sig)
formulas <- list(mu.fo = ~ x1, sigma.fo = ~ x2)
data <- data.frame(y = y, x1 = x1, x2=x2)



x_data <- runif(1000, 1, 5)
lambda1 <- exp(2.5 + 2 * x_data)
n <- 1000
y_data <- numeric()
for (i in 1:length(x_data)) {
        f <- function(x) {((lambda1[i]^2)+x-2*lambda1[i]) * exp(-x/lambda1[i]) / ((lambda1[i]^2) * (lambda1[i]-1))}

        Fa <- function(x) {integrate(f,0,x)$value}
        Fa <- Vectorize(Fa)

        F.inv <- function(y){uniroot(function(x){Fa(x)-y},interval=c(0,1), extendInt = "upX")$root}

        F.inv <- Vectorize(F.inv)


        Y <- runif(1,0,1)   # random sample from U[0,1]
        y_data[i] <- F.inv(Y)

}
data <- data.frame(y = y_data, x = x_data)
formulas <- list(loc.lambda = ~ x)


lambda1 <- 2.5
n <- 1000

f <- function(x) {((lambda1^2)+x-2*lambda1) * exp(-x/lambda1) / ((lambda1^2) * (lambda1-1))}

Fa <- function(x) {integrate(f,0,x)$value}
Fa <- Vectorize(Fa)

F.inv <- function(y){uniroot(function(x){Fa(x)-y},interval=c(0,1), extendInt = "upX")$root}

F.inv <- Vectorize(F.inv)

set.seed(499)
Y <- runif(n,0,1)   # random sample from U[0,1]
x <- F.inv(Y)




x <- runif(1000, 0, 3)
beta <- exp(2.5 - 2 * x)
concentration <- 2
y <- rgamma(1000, shape = concentration, scale = 1/beta)
data <- data.frame(y=y, x=x)
formulas <- list(rate.fo = ~ x)


x1 <- rep(1, 500)
x2 <- runif(500, 0, 30)
x3 <- runif(500, 0, 15)
x4 <- runif(500, 10, 20)
mui <- 15 + 2*x2 + 3*x3
alphai <- exp(0.2 + 0.1*x2 + 0.3*x4)
Y <- rgamma(500, shape=alphai, scale=mui/alphai)


x1 <- runif(1000, 0, 15)
x2 <- runif(1000, 10, 20)
beta <- exp(-3 + 9*x1 - 2*x2)
#beta <- 15
y <- rexp(1000, rate = beta)
data <- data.frame(y=y, x1=x1, x2=x2)
formulas <- list(rate.fo = ~ x1 + x2)

x1 <- runif(1000, 0, 1)
shape1 <- exp(-3 + 9*x1)
y <- rbeta(1000, shape1 = shape1, shape2 = 2)
data <- data.frame(y=y, x1=x1)
formulas <- list(concentration1.fo = ~ x1)

df <- 3
x <- rt(1000, df=df)


x <- runif(1000, 3, 10)



x <- rmultinom(10, size = 12, prob = c(0.1,0.2,0.8))

