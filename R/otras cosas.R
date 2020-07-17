call <- match.call()
limit <- call[[2]]


hello <- function(x, y){
        call <- match.call()
        limit <- call[[3]]
        l <- as.character(limit)
        return(l)
}

hello(1, 2)
call <- match.call()
call[[2]]

formulas <- list(sd.fo = ~ x, mean.fo = ~ x)
formulas_corrector <- stringr::str_extract(as.character(formulas), "~.+")
formulas_tmp <- as.list(formulas_corrector)
fos_bind <- function(formula, response){
        paste(response, paste(formula, collapse = " "))
}
Y <- rnorm(100, 0, 1)
fos_mat_char <- lapply(formulas_tmp, fos_bind, response = Y)
fos_mat <- lapply(fos_mat_char, as.formula)
list_mfs <- lapply(fos_mat, model.frame, data = data1)
#data <- as.data.frame(matrix(1, nrow = 100, ncol = 2))
#colnames(data) <- c("y", "x")
data1 <- NULL
if ( is.null(data1) ){
        data_reg <- as.data.frame(list_mfs)
        var_names <- as.character(unlist(sapply(list_mfs, names)))
        names(data_reg) <- var_names
        data_reg <- as.data.frame(data_reg[,unique(var_names)])
        names(data_reg) <- unique(var_names)
        data <- data_reg
}
