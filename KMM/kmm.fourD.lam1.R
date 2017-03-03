source('kmm.init.R')
library(foreach)
library(doParallel)

for (j in c(1:4)){
    lams <- rep(0.2,4)
    lams[j] <- 0.4
    lams_name <- paste0('lam1_',j)
#    lams <- c(0.2,0.2,0.2,0.4)
#    lams_name <- 'lam1_4'


    no_cores <- 10
    cl<-makeCluster(no_cores)
    registerDoParallel(cl)

    #==========
    # Load Data
    foreach(split = 1:10)  %dopar%  {
        source('kmm.init.R')

        source_types <- c('K'="kitchen",'D'="dvd",'B'="books",'E'="electronics")
        target_types <- c('K'="kitchen",'D'="dvd",'B'="books",'E'="electronics")


        datadir <- paste0('../data/exp',split,'/')
        train_sample_rate <- 0.25
        test_sample_rate <- 0.5

        # load training data
        X.train <- c()
        Y.train <- c()
        #sample_rate <- 1/length(source_types)
        for(type in source_types){
            loaded <- load_data(datadir, 'train',type)
            idx <- seq_len(train_sample_rate*length(loaded$Y))
            X.train <- rbind(X.train, loaded$X[idx,])
            Y.train <- c(Y.train, loaded$Y[idx])
        }


        X.test <- c()
        Y.test <- c()
        for(i in seq_along(target_types)){
            type <- target_types[i]
            loaded <- load_data(datadir, 'test',type)
            idx <- seq_len(test_sample_rate*length(loaded$Y)*lams[i])
            X.test <- rbind(X.test, loaded$X[idx,])
            Y.test <- c(Y.test, loaded$Y[idx])
        }

        #==========
        ### Compute the solution with KMM 
        sigma <- 0.00078 
        m <- nrow(X.train)
        n <- nrow(X.test)
        K <- form.K(X.train,X.test, sigma)

        B <- 1000
        kmm.u <- kmm(K,m,n,B)
        kmm.u <- kmm.u/sum(kmm.u)

        cat(split,':', paste0(names(target_types),collapse = ''),'done')

        filename <- paste0(paste0('split_',split,'_'),
                           '4sources_lams_',lams_name,'_',
                           paste0('sigma_',sigma), '.KMM_sol.csv')
        write.table(kmm.u,file = filename,col.names = FALSE, row.names = FALSE,quote= FALSE,sep = ",")
    }

    stopCluster(cl)


}


