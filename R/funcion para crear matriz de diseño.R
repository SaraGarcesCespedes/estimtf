
y_dist <- y ~ dnorm
npar <- 2
n <- 1000
x <- runif(n = n, -5, 6)
y <- rnorm(n = n, mean = -2 + 3 * x, sd = exp(1 + 0.3* x))
data <- data.frame(y = y, x = x)
par_names <- c("sd", "mean")
# It does not matter the order of distribution paramters
formulas <- list(sd.fo = ~ x, mean.fo = ~ x)
model.matrix.MLreg(formulas, norm_data, y_dist, npar, par_names)

matrixes <- function(j, formulas, model_frames){
        do.call(what = "model.matrix",
                args = list(object = as.formula(formulas[[j]]),
                            data = model_frames[[j]]))
}
fos_bind <- function(formula, response){
        paste(response, paste(formula, collapse = " "))
}
model.matrix.MLreg <- function(formulas, data, y_dist, np, par_names){

        # Errors in formulas
        if (!any(lapply(formulas, class) == "formula")){
                stop("All elements in argument 'formulas' must be of class formula")
        }

        # Number of formulas (one formula for each parameter)
        nfos <- length(formulas)

        if (nfos != npar) stop(paste0("Distribution defined for response ",
                                      "variable has ", npar, " parameters to be estimated. ",
                                      "Each parameter must have its own formula"))

        # Response variable
        #if (!inherits(y_dist, "formula")) stop(paste0("Expression in 'y_dist' ",
         #                                               "must be of class 'formula"))
        #if (length(y_dist) != 3) stop(paste0("Expression in 'y_dist' ",
         #                                      "must be a formula of the form ",
          #                                     "'response ~ distribution' or ",
           #                                    "'Surv(response, status) ~ distribution'"))

        #Y <- all.vars(y_dist)[1] #Surv_transform(y_dist = y_dist)

        # Extract the right side of formulas
        formulas_corrector <- stringr::str_extract(as.character(formulas), "~.+")
        formulas_tmp <- as.list(formulas_corrector)
        names(formulas_tmp) <- par_names

        # Variables
        fos_mat_char <- lapply(formulas_tmp, fos_bind, response = y)
        fos_mat <- lapply(fos_mat_char, as.formula)
        list_mfs <- lapply(fos_mat, model.frame, data = norm_data)
        if ( is.null(data) ){
                data_reg <- as.data.frame(list_mfs)
                var_names <- as.character(unlist(sapply(list_mfs, names)))
                names(data_reg) <- var_names
                data_reg <- as.data.frame(data_reg[,unique(var_names)])
                names(data_reg) <- unique(var_names)
                data <- data_reg
        }
        response <- model.frame(fos_mat[[1]], data = data)[, 1]

        # Censorship status
        # cens <- Surv_transform(y_dist = y_dist, data = data)

        # Formulas for 'model.frame'
        mtrxs <- lapply(X = 1:nfos, FUN = matrixes, formulas = fos_mat,
                        model_frames = list_mfs)

        names(mtrxs) <- names(fos_mat)
        mtrxs$y <- response
        # mtrxs$status <- cens[,2:ncol(cens)]
        mtrxs$data_reg <- data
        return(mtrxs)
}
