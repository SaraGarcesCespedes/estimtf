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
fos_mat_char <- lapply(formulas_tmp, fos_bind, response = Y)
fos_mat <- lapply(fos_mat_char, as.formula)
list_mfs <- lapply(fos_mat, model.frame, data = data)
data <- as.data.frame(matrix(1, nrow = 100, ncol = 2))
colnames(data) <- c("y", "x")
