#------------------------------------------------------------------------
# Parameters names from R to TF -----------------------------------------
#------------------------------------------------------------------------
parameter_name_tf <- function(parameter, distribution) {

        list_dist <- c("Poisson", "LogNormal")
        if (distribution %in% list_dist) {
                listparam <- list(lambda = "rate",
                                  meanlog = "log",
                                  sdlog = "scale")
        } else if (distribution == "FWE") {
                listparam <- list(mu = "mu",
                                  sigma = "sigma")
        } else {
                listparam <- list(mean = "loc",
                                  sd = "scale",
                                  shape = "concentration",
                                  rate = "rate",
                                  scale = "scale",
                                  shape1 = "concentration1",
                                  shape2 = "concentration0",
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
                listparam <- list(rate = "lambda",
                                  log = "meanlog",
                                  scale = "sdlog")
        } else if (distribution == "FWE") {
                listparam <- list(mu = "mu",
                                  sigma = "sigma")
        } else {
                listparam <- list(loc = "mean",
                                  scale = "sd",
                                  concentration = "shape",
                                  rate = "rate",
                                  scale = "scale",
                                  concentration1 = "shape1",
                                  concentration0 = "shape2",
                                  total_count = "size",
                                  probs = "prob")
        }

        return(listparam[[parameter]])
}
