#Load libraries
library(kernlab)
library(quadprog)
library(corpcor)

form.K <- function
(train, 
test,
sigma = 0.00078 
 ){
  A <- rbind(train,test)
  #K <- exp(-as.matrix(dist(A)^2)/(2*sigma^2))
  K <- exp(-as.matrix(dist(A)^2)*(sigma))
  return (K)
}


threshold.red <- function ## ridge regression with weights
(train.s, # train feature
 weights,# train sample weights
 targets,# train label
 test.s,# test feature
 test.targets, # test label
 K,# kernel matrix: train x train
 K.test.train,# kernel matrix: test x train
 size=pp, # sample training set for ridge regression
 eps.cutoff = 10^(-10),
 lambda = 0.0009765625
 ){

  m=dim(train.s)[1]
  weights[weights<0]=0
  sel=weights>eps.cutoff
  subset=rep(FALSE,m)

  ## Maybe we want to do modeling with a subset of the training points
  set.seed(size)
  subs=sample(1:m,size)
  subset[subs]=TRUE
  #print(weights)
  m.eff=sum(sel&subset)
  #print(paste("m.eff(threshold)=", m.eff))
  #if(m.eff<1){ print(weights)}

  # weight (subset of) training points 
  K.train=K[1:m,1:m][sel&subset,sel&subset]
  sqrt.weights=sqrt(weights)[sel&subset]
  K.train.w = K.train * (sqrt.weights%*%t(sqrt.weights))
  targets.w = targets[sel&subset] * sqrt.weights

  # alpha: representer coefficients 
  alpha=solve(K.train.w + m.eff*diag(lambda,m.eff))%*%targets.w

  # fit1: predict on training
  fit1=K[1:m,1:m][,sel&subset] %*% (as.matrix(sqrt.weights,ncol=1) *alpha)
  assign("fit1",fit1,envir = .GlobalEnv)
  #print(paste("TRAINING error",mean((fit1-targets)^2)))

  # fit2: predict on testing
  fit2=K.test.train[,sel&subset] %*% (as.matrix(sqrt.weights,ncol=1) *alpha)

  assign("fit2",fit2,envir = .GlobalEnv)

  return (list(mean((fit2 - test.targets)^2),mean((fit1-targets)^2)))
}

kmm <- function
## min_b (- kappa ^T b + 1/2 b^T K.tr b)
## s.t. 	0 <= b_i <= B, i=1,...,m
## 	n(1-eps) <= sum(b_i) <= n(1+eps)	
(K,# (train, adapt) x (train,adapt)
 m, #  train size
 n, #  adapt size
 B, # upper bound on weights
 eps.kmm = 1e-6
 ){
    eps.kmm = sqrt(m)/(sqrt(m)-1)

    print("entering kmm")
    # train kernel
    K.tr=K[1:m,1:m]+diag(eps.kmm,m)

    # build QP 
    kappa=apply(K[(m+1):(m+n),1:m],2,sum)
    A=cbind(diag(1,m),diag(-1,m),rep(1,m),rep(-1,m))
    bvec=c(rep(0,m),rep(-B,m),(1-eps.kmm)*n,-(eps.kmm+1)*n)
    kmm=solve.QP(K.tr,kappa,A,bvec)

    # return
    if(sum(is.na(kmm$solution))>0){kmm$solution=rep(1/m,m)}
    return (kmm$solution)
}


load_data <- function
(datadir, 
 dset = c('train','test')[1],
 type # source domains
){

  Xfile <- paste0(datadir,type,'.X.',dset,'.csv')
  labelfile <- paste0(datadir,type,'.',dset,'.labels')

  X <- read.csv(Xfile, header = FALSE)
  Y <- read.csv(labelfile, header = FALSE)
  Y <- c(Y[,1])

  return(list(X=X,Y=Y))
}
