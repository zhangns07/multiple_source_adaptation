#==========
# A helper script that compute mean+/- st in latex table format
#==========
testmixtures <- c('K','D','B','E','KD','KB','KE','DB','DE','BE','DBE','KBE','KDE','KDB','KDBE')
methods <- c('wz','wlam','cvx','unif','K','D','B','E')
methodstokeep <- c(5:8,4,1)
nc <- length(testmixtures)
nr <- length(methodstokeep)

RET = array(rep(0,nr*nc*10),c(nr,nc,10))

for (i in seq_len(10)){
    ddt = read.csv(paste0('kitchen_dvd_books_electronics2gram_2mino_',i,'_ontest.csv'),header = FALSE)
    colnames(ddt) <- methods
    rownames(ddt) <- testmixtures
    ddt2 <- t(ddt[,methodstokeep])
    RET[,,i] = as.matrix(ddt2)
}

MSE = apply(RET,c(1,2),mean)
MSEsd = apply(RET,c(1,2),sd)

MSEbf = matrix(rep("",nr,nc),nr,nc)
minidx = apply(MSE,2,which.min)
for (j in seq_len(ncol(MSEbf))){
    MSEbf[minidx[j],j] <- '\\bf'
}

preds <- paste0('\\texttt{',c('K','D','B','E','unif','DW'),'}')
colstokeep <- c(1:4,10,15,5:9)
#colstokeep <- c(1:4,5,10,11:15)
#colstokeep <- c(6:9)

for (i in seq_len(nrow(MSE))){

    cat(paste0(c(preds[i],
                 paste0( '{',
                        MSEbf[i,colstokeep],
                        sprintf('%.2f',MSE[i,colstokeep]),'$\\pm$', 
                  sprintf('%.2f',MSEsd[i,colstokeep]),'}'
                  )),collapse = "\t&\t"))
    cat('\\\\')
    cat('\n')
}


