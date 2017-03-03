#----------
# X in 2d
library(data.table)
library(mvtnorm)
library(ggplot2)

Nsample = 1000
Niter = 9
set.seed(0)


for (iter in seq_len(Niter)){
  input = data.table(x1 = runif(Nsample,-2,2), x2 = runif(Nsample,-2,2))

  # density: multivariate gaussian with centers in four quadrants.
  input[,p1:=dmvnorm(c(x1,x2), c(1,1)),by=rownames(input)]
  input[,p2:=dmvnorm(c(x1,x2), c(-1,1)),by=rownames(input)]
  input[,p3:=dmvnorm(c(x1,x2), c(-1,-1)),by=rownames(input)]
  input[,p4:=dmvnorm(c(x1,x2), c(1,-1)),by=rownames(input)]

  # domain two: mixture of (1,2,3) and (2,3,4) quadrants.
  input[,D1:=(p1+p2+p3)]; input[,D1:=D1/sum(D1)]
  input[,D2:=(p2+p3+p4)]; input[,D2:=D2/sum(D2)]
  input[,U:=1/Nsample]
  input[,label:=x1^2+x2^2]

  # linear regression
  model1 <- lm( label  ~ x1+x2, data = input, weights = D1)
  input[,h1:=predict(model1)]
  model2 <- lm( label ~ x1+x2, data = input, weights = D2)
  input[,h2:=predict(model2)]

  write.table(input[,list(D1,D2,h1,h2,label)],file = paste0('iter_',iter,'.csv'),sep = ',',col.names = FALSE, row.names = FALSE, quote = FALSE)
}


