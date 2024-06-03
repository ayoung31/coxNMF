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
