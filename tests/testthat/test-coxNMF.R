test_that("update_H R and cpp give same result", {
  n=10
  p=15
  k=2

  X=matrix(runif(p*n,0,5),nrow=p,ncol=n)
  M=matrix(1,nrow=p,ncol=n)
  W=matrix(runif(p*k,0,5),nrow=p,ncol=k)
  H=matrix(runif(k*n,0,5),nrow=k,ncol=n)
  beta=rep(1,k)
  y=runif(n,0,5)
  delta=rbinom(n,1,.5)
  alpha=.5

  fit1=update_H(X,M,W,beta,H,y,delta,alpha,FALSE)
  fit2=update_H_cpp(X,M,W,beta,H,y,delta,alpha,FALSE)

  expect_equal(fit1,fit2)
})

test_that("WtX update_H R and cpp give same result", {
  n=10
  p=15
  k=2

  X=matrix(runif(p*n,0,5),nrow=p,ncol=n)
  M=matrix(1,nrow=p,ncol=n)
  W=matrix(runif(p*k,0,5),nrow=p,ncol=k)
  H=matrix(runif(k*n,0,5),nrow=k,ncol=n)
  beta=rep(1,k)
  y=runif(n,0,5)
  delta=rbinom(n,1,.5)
  alpha=.5

  fit1=update_H(X,M,W,beta,H,y,delta,alpha,TRUE)
  fit2=update_H_cpp(X,M,W,beta,H,y,delta,alpha,TRUE)

  expect_equal(fit1,fit2)
})

test_that("update_W R and cpp give same result column normalization", {
  n=10
  p=15
  k=2

  X=matrix(runif(p*n,0,5),nrow=p,ncol=n)
  M=matrix(1,nrow=p,ncol=n)
  W=matrix(runif(p*k,0,5),nrow=p,ncol=k)
  H=matrix(runif(k*n,0,5),nrow=k,ncol=n)
  beta=rep(1,k)
  y=runif(n,0,5)
  delta=rbinom(n,1,.5)
  alpha=.5

  fit1=update_W(X,M,H,W,beta,y,delta,alpha,FALSE,2)
  fit2=update_W_cpp(X,M,H,W,beta,y,delta,alpha,FALSE,2)

  expect_equal(fit1,fit2)
})

test_that("update_W R and cpp give same result row normalization", {
  n=10
  p=15
  k=2

  X=matrix(runif(p*n,0,5),nrow=p,ncol=n)
  M=matrix(1,nrow=p,ncol=n)
  W=matrix(runif(p*k,0,5),nrow=p,ncol=k)
  H=matrix(runif(k*n,0,5),nrow=k,ncol=n)
  beta=rep(1,k)
  y=runif(n,0,5)
  delta=rbinom(n,1,.5)
  alpha=.5

  fit1=update_W(X,M,H,W,beta,y,delta,alpha,FALSE,1)
  fit2=update_W_cpp(X,M,H,W,beta,y,delta,alpha,FALSE,1)

  expect_equal(fit1,fit2)
})

test_that("calc_loss gives same results in R and cpp", {
  n=10
  p=15
  k=2
  
  X=matrix(runif(p*n,0,5),nrow=p,ncol=n)
  M=matrix(1,nrow=p,ncol=n)
  W=matrix(runif(p*k,0,5),nrow=p,ncol=k)
  H=matrix(runif(k*n,0,5),nrow=k,ncol=n)
  beta=rep(1,k)
  y=runif(n,0,5)
  delta=rbinom(n,1,.5)
  alpha=.5
  
  fit1=calc_loss(X,M,W,H,beta,alpha,y,delta,.1,.5,FALSE)
  fit2=calc_loss_cpp(X,M,W,H,beta,alpha,y,delta,.1,.5,FALSE)
  
  expect_equal(fit1,fit2)
})

test_that("optimize_loss_cpp function produces same result as optimize_loss function", {
  n=10
  p=15
  k=2
  set.seed(1)
  X=matrix(runif(p*n,0,5),nrow=p,ncol=n)
  M=matrix(1,nrow=p,ncol=n)
  W=matrix(runif(p*k,0,5),nrow=p,ncol=k)
  H=matrix(runif(k*n,0,5),nrow=k,ncol=n)
  beta=rep(1,k)
  y=runif(n,0,5)
  delta=rbinom(n,1,.5)
  alpha=.5
  
  fit1=optimize_loss(X,M,H,W,beta,y,delta,alpha,.1,.5,1e-6,1000,FALSE,FALSE,2)
  fit2=optimize_loss_cpp(X,M,H,W,beta,y,delta,alpha,.1,.5,1e-6,1000,FALSE,FALSE,2)
  
  expect_equal(fit1,fit2)
})