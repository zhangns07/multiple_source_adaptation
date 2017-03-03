source('kmm.init.R')

#==========
# Load Data
seed <- 0
B <- 1000
train_sample_rate <- 0.5
test_sample_rate <- 0.5

#source_types <- c('K'="kitchen",'B'="books")
#target_types <- c('K'="kitchen",'B'="books")
source_types <- c('D'="dvd",'E'="electronics")
target_types <- c('D'="dvd",'E'="electronics")

split <- 2
datadir <- paste0('../data/exp',split,'/')

# load training data
X.train <- c()
Y.train <- c()
for(type in source_types){
    loaded <- load_data(datadir, 'train',type)
    idx <- seq_len(train_sample_rate*length(loaded$Y))
    X.train <- rbind(X.train, loaded$X[idx,])
    Y.train <- c(Y.train, loaded$Y[idx])
}


#lam <- 0.4 #target mixture
for (lam in c(0:10)/10){ # window 1
    # load testing (adapt) data
    LAM <- c(lam,1-lam)
    X.test <- c()
    Y.test <- c()
    for(i in seq_along(target_types)){
        type <- target_types[i]
        loaded <- load_data(datadir, 'test',type)
        n <- length(loaded$Y)
        idx <- seq_len(n*LAM[i]*test_sample_rate)
        X.test <- rbind(X.test, loaded$X[idx,])
        Y.test <- c(Y.test, loaded$Y[idx])
    }

    #==========
    ### Compute the solution with KMM 
    sigma <- 0.00078 
    m <- nrow(X.train)
    n <- nrow(X.test)
    K <- form.K(X.train,X.test, sigma)

    kmm.u <- kmm(K,m,n,B)
    kmm.u <- kmm.u/sum(kmm.u)

    filename <- paste0(paste0('split_',split,'_'),
                       'source_',paste0(names(source_types),collapse = ''),'_',
                       'targetmix_',lam,'_',
                       paste0('sigma_',sigma), '.KMM_sol.csv')

    write.table(kmm.u,file = filename,col.names = FALSE, row.names = FALSE,quote= FALSE,sep = ",")
}
