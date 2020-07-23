h <- lapply(1:np, FUN = hola)
h <- replxcate(1:np, expr = hola(x))
lapply(1:np, FUN = hola(x))
nbetas <- param <- vector(mode = "list", length = np)
names(nbetas) <- names(param) <- names(design_matrix)[1:np]
totalbetas <- sum(as.numeric(unlist(sapply(design_matrix[1:np], ncol))))
regparam <- vector(mode = "list", length = totalbetas)
t <- 0
hola <- function(x) {
        sum <- 0
        nbetas[[x]] <- sum(as.numeric(unlist(sapply(design_matrix[x], ncol))))
        multparam <- vector(mode = "list", length = nbetas[[x]])
        x_data <- eval(parse(text = paste("design_matrix$", names(nbetas)[x], sep = "")))
        bnum <- rep(0:(nbetas[[x]]-1))
        for (j in 1:nbetas[[x]]){
                regparam[[j + t]] <- assign(paste0("beta", bnum[j], names(nbetas)[x]),
                                            tf$Variable(initparam[[names(nbetas)[x]]],
                                                        dtype = tf$float32),
                                            envir = .GlobalEnv)
                names(regparam)[j + t] <- paste0("beta", bnum[j], names(nbetas)[x])
                multparam[[j]] <- tf$multiply(x_data[, j], regparam[[j + t]])
                sum <- sum + multparam[[j]]
        }
        sum <- link(link_function, sum, names(nbetas)[x])
        param[[x]] <- assign(names(nbetas)[x], sum, envir = .GlobalEnv)
        t <- t + nbetas[[x]]
        return(regparam)
}



hola2 <- function(y, bnum, x_data, multparam, sum) {
        #bnum <- rep(0:(nbetas[[x]]-1))
        regparam[[y + t]] <- assign(paste0("beta", bnum[y], names(nbetas)[x]),
                                            tf$Variable(initparam[[names(nbetas)[x]]],
                                                        dtype = tf$float32),
                                    envir = .GlobalEnv)
        names(regparam)[y + t] <- paste0("beta", bnum[y], names(nbetas)[x])
        multparam[[y]] <- tf$multiply(x_data[, y], regparam[[y + t]])
        sum <- sum + multparam[[y]]
        return(sum)
}



hola2 <- function(y, bnum, x_data, multparam, sum) {
        #bnum <- rep(0:(nbetas[[x]]-1))
        regparam[[y + t]] <- assign(paste0("beta", bnum[y], names(nbetas)[x]),
                                    tf$Variable(initparam[[names(nbetas)[x]]],
                                                dtype = tf$float32),
                                    envir = .GlobalEnv)
        names(regparam)[y + t] <- paste0("beta", bnum[y], names(nbetas)[x])
        multparam[[y]] <- tf$multiply(x_data[, y], regparam[[y + t]])
        sum <- sum + multparam[[y]]
        return(sum)
}

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

x <- 1
regparam <- vector(mode = "list", length = 1)

j <- function(x) {
        regparam[[x]] <- assign(paste0("beta"),
                                    tf$Variable(1.0,
                                                dtype = tf$float32), envir = .GlobalEnv)

}

a <- lapply(1, FUN = j)

