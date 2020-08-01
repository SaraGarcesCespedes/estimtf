# Otras funciones (PAQUETE ESTIMATION TOOLS)
matrixes <- function(j, formulas, model_frames){
        do.call(what = "model.matrix",
                args = list(object = as.formula(formulas[[j]]),
                            data = model_frames[[j]]))
}

fos_bind <- function(formula, response){
        paste(response, paste(formula, collapse = " "))
}



model.matrix.MLreg <- function(formulas, data, ydist, np, par_names){

        # Errors in formulas
        if (!any(lapply(formulas, class) == "formula")){
                stop("All elements in argument 'formulas' must be of class formula")
        }

        # Number of formulas (one formula for each parameter)
        nfos <- length(formulas)

        if (nfos != np) stop(paste0("Distribution defined for response ",
                                    "variable has ", npar, " parameters to be estimated. ",
                                    "Each parameter must have its own formula"))

        # Response variable
        if (!inherits(ydist, "formula")) stop(paste0("Expression in 'y_dist' ",
                                                     "must be of class 'formula"))
        if (length(ydist) != 3) stop(paste0("Expression in 'y_dist' ",
                                            "must be a formula of the form ",
                                            "'response ~ distribution' or ",
                                            "'Surv(response, status) ~ distribution'"))

        Y <- all.vars(ydist)[1] #Surv_transform(y_dist = y_dist)

        # Extract the right side of formulas
        formulas_corrector <- stringr::str_extract(as.character(formulas), "~.+")
        formulas_tmp <- as.list(formulas_corrector)
        names(formulas_tmp) <- par_names

        # Variables
        fos_mat_char <- lapply(formulas_tmp, fos_bind, response = Y)
        fos_mat <- lapply(fos_mat_char, as.formula)
        list_mfs <- lapply(fos_mat, model.frame, data = data)
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
