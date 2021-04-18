#' @title predict.MLEtf function
#'
#' @description Function to produce result summaries of the estimates of parameters from statistical
#' distributions using \code{\link{dist_estimtf2}} or parameters from regression models using
#' \code{\link{reg_estimtf}}.
#'
#' @author Sara Garces Cespedes
#'
#' @param object an object of class \code{MLEtf} for which a summary is desired.
#' @param newdata an optional data frame in which to look variables with which to predict. If \code{newdata} is
#' missing or \code{NULL}, the fitted values are used.
#' @param ... additional arguments affecting the summary produced.
#'
#' @return The output from
#'
#' @details \code{predict.MLEtf} COMPLETAR
#'
#' @importFrom stats delete.response
#'
#' @rdname predict.MLEtf
#' @export
#------------------------------------------------------------------------
# Predict function ------------------------------------------------------
#------------------------------------------------------------------------
predict.MLEtf <- function(object, newdata = NULL, ...) {

        if (!inherits(object, "MLEtf")) {
                warning("Object is not class MLEtf")
        }

        # error con newdata
        if (missing(newdata) | is.null(newdata)) {
                data <- object$data
        } else {
                if (!is.data.frame(newdata)) {
                        stop("newdata must be a data frame")
                } else {
                        data <- newdata
                }
        }

        formula <- object$formula

        tt <- object$tt

        Terms <- delete.response(tt)

        xlevels <- object$xlevels

        data_reg_predict <- model.frame(Terms, data = data, xlev = xlevels)

        dsg_matrix_predict <- model.matrix(object = Terms, data = data_reg_predict)

        coeff_values <- object$outputs$estimates

        if (object$distribution == "Normal") {
                coeff_values <- coeff_values[, -ncol(coeff_values)]
        }

        # estimate Y
        Y <- dsg_matrix_predict %*% coeff_values
        Y <- as.numeric(Y)



        return(Y)
}
