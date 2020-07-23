
#REGESTIMTF

for (i in 1:length(link_function)) {
        if (!(link_function[[i]] %in% lfunctions)) {
                stop(paste0("Unidentified link function Select one of the link functions included in the \n",
                            " following list: ", paste0(lfunctions, collapse = ", ")))
        }
}

link_function <- list(loc = "log", scale = "lo")
verifylink <- lapply(1:length(link_function), FUN = function(x) {
        if (!(link_function[[x]] %in% lfunctions)) {
                stop(paste0("Unidentified link function Select one of the link functions included in the \n",
                            " following list: ", paste0(lfunctions, collapse = ", ")))
        }
})


for (i in 1:np) initparam[[i]] <- 0.0 #SEGURAMENTE SE PUEDE HACER MAS EFICIENTE

initparam <- lapply(1:np, FUN = function(i) initparam[[i]] <- 0)



for (i in 1:length(hyperparameters)) hyperparameters[[i]] <- ifelse(splitarg[[i]][2] == "True" | splitarg[[i]][2] == "False", splitarg[[i]][2], as.numeric(splitarg[[i]][2])) #SE PUEDE HACER MAS EFICIENTE?

hyperparameters <- lapply(1:length(hyperparameters),
                          FUN = function(i) hyperparameters[[i]] <- ifelse(splitarg[[i]][2] == "True" | splitarg[[i]][2] == "False",
                                                                           splitarg[[i]][2], as.numeric(splitarg[[i]][2])))


#DIST_ESTIMTF

for (i in 1:length(np)) initparam[[i]] <- ifelse(dist == "Instantaneous Failures" | dist == "Poisson", 2.0, 0.0)

initparam <- lapply(1:np,
                    FUN = function(i) initparam[[i]] <- ifelse(dist == "Instantaneous Failures" | dist == "Poisson", 2.0, 0.0))


for (i in 1:length(hyperparameters)) hyperparameters[[i]] <- ifelse(splitarg[[i]][2] == "True" | splitarg[[i]][2] == "False", splitarg[[i]][2], as.numeric(splitarg[[i]][2])) #SE PUEDE HACER MAS EFICIENTE?

hyperparameters <- lapply(1:length(hyperparameters),
                          FUN = function(i) hyperparameters[[i]] <- ifelse(splitarg[[i]][2] == "True" | splitarg[[i]][2] == "False",
                                                                           splitarg[[i]][2], as.numeric(splitarg[[i]][2])))


#AUXILIARYFUNMOD1
#disable eager
for (i in 1:np) {
        var_list[[i]] <- assign(names(initparam)[i], tf$Variable(initparam[[i]], dtype = tf$float32, name = names(initparam)[i]))
}


var_list <- lapply(1:np, FUN = function(i) var_list[[i]] <- assign(names(initparam)[i],
                                                                   tf$Variable(initparam[[i]],
                                                                               dtype = tf$float32,
                                                                               name = names(initparam)[i])))


for (i in 1:np) new_list[[i]] <- var_list[[i]]
new_list <- lapply(1:np, FUN = function(i) new_list[[i]] <- var_list[[i]])



for (i in 1:np) {
        objvariables[[i]] <- as.numeric(sess$run(new_list[[i]]))
        #objvariables[[i]] <- as.numeric(objvariables[[i]])
        itergrads[[i]] <- as.numeric(sess$run(grads, feed_dict = fd)[[i]])
}

objvariables <- lapply(1:np, FUN = function(i) objvariables[[i]] <- as.numeric(sess$run(new_list[[i]])))
itergrads <- lapply(1:np, FUN = function(i) itergrads[[i]] <- as.numeric(sess$run(grads, feed_dict = fd)[[i]]))


for(i in 1:np) hesslist[[i]] <- tf$gradients(grads[[i]], new_list)
hesslist <- lapply(1:np, FUN = function(i) hesslist[[i]] <- tf$gradients(grads[[i]], new_list))


for (i in 1:np) stderror[[i]] <- diagvarcov[i] #ESTO PUEDE SER MAS EFICIENTE
stderror <- lapply(1:np, FUN = function(i) stderror[[i]] <- diagvarcov[i])


for (j in 1:np) {
        gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[j]]))
        parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[j]]))
        namesgradients <- cbind(namesgradients, paste0("Gradients ", names(var_list)[j]))
}

gradientsfinal <- sapply(1:np, function(i) cbind(gradientsfinal, as.numeric(gradients[[i]])))
parametersfinal <- sapply(1:np, function(i) cbind(parametersfinal, as.numeric(parameters[[i]])))
namesgradients <- sapply(1:np, function(i) cbind(namesgradients, paste0("Gradients ", names(var_list)[i])))

#EGAER EXCECUTION
for (i in 1:np) {
        var_list[[i]] <- assign(names(initparam)[i], tf$Variable(initparam[[i]], dtype = tf$float32, name = names(initparam)[i]))
}

var_list <- lapply(1:np, FUN = function(i) var_list[[i]] <- assign(names(initparam)[i],
                                                                   tf$Variable(initparam[[i]],
                                                                               dtype = tf$float32,
                                                                               name = names(initparam)[i])))


for (i in 1:np) new_list[[i]] <- var_list[[i]]
new_list <- lapply(1:np, FUN = function(i) new_list[[i]] <- var_list[[i]])

for(i in 1:np) hesslist[[i]] <- tape$gradient(grads[[i]], new_list)
hesslist <- lapply(1:np, FUN = function(i) hesslist[[i]] <- tape$gradient(grads[[i]], new_list))

for (i in 1:np) {
        objvariables[[i]] <- as.numeric(get(names(var_list)[i]))
        gradients[[step]][[i]] <- as.numeric(gradients[[step]][[i]])
}

objvariables <- lapply(1:np, objvariables[[i]] <- as.numeric(get(names(var_list)[i])))
gradients[[step]] <- lapply(1:np, gradients[[step]][[i]] <- as.numeric(gradients[[step]][[i]]))


for (i in 1:np) stderror[[i]] <- diagvarcov[i] #ESTO PUEDE SER MAS EFICIENTE
stderror <- lapply(1:np, FUN = function(i) stderror[[i]] <- diagvarcov[i])


for (j in 1:np) {
        gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[j]]))
        parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[j]]))
        namesgradients <- cbind(namesgradients, paste0("Gradients ", names(var_list)[j]))
}

gradientsfinal <- sapply(1:np, function(i) cbind(gradientsfinal, as.numeric(gradients[[i]])))
parametersfinal <- sapply(1:np, function(i) cbind(parametersfinal, as.numeric(parameters[[i]])))
namesgradients <- sapply(1:np, function(i) cbind(namesgradients, paste0("Gradients ", names(var_list)[j])))

#AUXILIARYFUNMOD2
#DISABLE EAGER
#falta una del inicio

for (i in 1:length(regparam)) {
        objvariables[[i]] <- as.numeric(sess$run(regparam[[i]]))
        itergrads[[i]] <- as.numeric(sess$run(grads, feed_dict = fd)[[i]])
}

objvariables <- lapply(1:length(regparam), FUN = function(i) objvariables[[i]] <- as.numeric(sess$run(regparam[[i]])))
itergrads <- lapply(1:length(regparam), FUN = function(i) itergrads[[i]] <- as.numeric(sess$run(grads, feed_dict = fd)[[i]]))



for(i in 1:length(regparam)) hesslist[[i]] <- tf$gradients(grads[[i]], regparam)
hesslist <- lapply(1:length(regparam), FUN = function(i) hesslist[[i]] <- tf$gradients(grads[[i]], regparam))

for (i in 1:length(regparam)) stderror[[i]] <- diagvarcov[i]
stderror <- lapply(1:length(regparam), FUN = function(i) stderror[[i]] <- diagvarcov[i])

for (j in 1:length(regparam)) {
        gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[j]]))
        parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[j]]))
        namesgradients <- cbind(namesgradients, paste0("Gradients ", names(regparam)[j]))
}

gradientsfinal <- sapply(1:length(regparam), function(i) gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[i]])))
parametersfinal <- sapply(1:length(regparam), function(i) parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[i]])))
namesgradients <- sapply(1:length(regparam), function(i) namesgradients <- cbind(namesgradients, paste0("Gradients ", names(regparam)[i])))



#eager execution

for (i in 1:length(regparam)) new_list[[i]] <- regparam[[i]]
new_list <- lapply(1:length(regparam), FUN = function(i) new_list[[i]] <- regparam[[i]])


for(i in 1:length(new_list)) hesslist[[i]] <- tape$gradient(grads[[i]], new_list)
hesslist <- lapply(1:length(new_list), FUN = function(i) hesslist[[i]] <- tape$gradient(grads[[i]], new_list))


for (i in 1:length(regparam)) {
        objvariables[[i]] <- as.numeric(get(names(regparam)[i]))
        gradients[[step]][[i]] <- as.numeric(gradients[[step]][[i]])
}

objvariables <- lapply(1:length(regparam), objvariables[[i]] <- as.numeric(get(names(regparam)[i])))
gradients[[step]] <- lapply(1:length(regparam), gradients[[step]][[i]] <- as.numeric(gradients[[step]][[i]]))


for (i in 1:length(new_list)) stderror[[i]] <- diagvarcov[i]
stderror <- lapply(1:length(new_list), FUN = function(i) stderror[[i]] <- diagvarcov[i])


for (j in 1:length(new_list)) {
        gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[j]]))
        parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[j]]))
        namesgradients <- cbind(namesgradients, paste0("Gradients ", names(regparam)[j]))
}

gradientsfinal <- sapply(1:length(new_list), function(i) gradientsfinal <- cbind(gradientsfinal, as.numeric(gradients[[i]])))
parametersfinal <- sapply(1:length(new_list), function(i) parametersfinal <- cbind(parametersfinal, as.numeric(parameters[[i]])))
namesgradients <- sapply(1:length(new_list), function(i) namesgradients <- cbind(namesgradients, paste0("Gradients ", names(regparam)[i])))



#comparisonreg

if (!is.null(fixparam)) for (i in 1:length(fixparam)) names(fixparam)[i] <- parametersr[[match(names(fixparam)[i], names(parametersr))]]
if (!is.null(initparam)) for (i in 1:length(initparam)) names(initparam)[i] <- parametersr[[match(names(initparam)[i], names(parametersr))]]
if (!is.null(link_function)) for (i in 1:length(link_function)) names(link_function)[i] <- parametersr[[match(names(link_function)[i], names(parametersr))]]

for (i in 1:length(link_function)) {
        link[[i + p]] <- names(link_function)[i]
        link[[i + p + 1]] <- paste0(link_function[[i]], "_link")
        names(link)[[i + p]] <- "over"
        names(link)[[i + p + 1]] <- "fun"
        p <- 1
}

