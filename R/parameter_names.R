#------------------------------------------------------------------------
# Parameters names from R to TF -----------------------------------------
#------------------------------------------------------------------------
parameter_name_tf <- function(parameter, distribution) {

        list_dist <- c("Poisson", "LogNormal")
        if (distribution %in% list_dist) {
                listparam <- list(lambda = "lambda",
                                  meanlog = "log",
                                  sdlog = "scale")
        } else if (distribution == "FWE") {
                listparam <- list(mu = "mu",
                                  sigma = "sigma")
        } else {
                listparam <- list(mean = "mean",
                                  sd = "sd",
                                  shape = "shape",
                                  rate = "rate",
                                  scale = "scale",
                                  shape1 = "shape1",
                                  shape2 = "shape2",
                                  size = "total_count",
                                  prob = "probs")
        }


        return(listparam[[parameter]])
}

#------------------------------------------------------------------------
# Parameters names from TF to R -----------------------------------------
#------------------------------------------------------------------------
parameter_name_R <- function(parameter, distribution) {

        list_dist <- c("Poisson", "LogNormal")
        if (distribution %in% list_dist) {
                listparam <- list(lambda = "lambda",
                                  log = "meanlog",
                                  scale = "sdlog")
        } else if (distribution == "FWE") {
                listparam <- list(mu = "mu",
                                  sigma = "sigma")
        } else {
                listparam <- list(mean = "mean",
                                  sd = "sd",
                                  shape = "shape",
                                  rate = "rate",
                                  scale = "scale",
                                  shape1 = "shape1",
                                  shape2 = "shape2",
                                  total_count = "size",
                                  probs = "prob")
        }

        return(listparam[[parameter]])
}
