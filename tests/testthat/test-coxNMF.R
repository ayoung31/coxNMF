test_that("update_H R and cpp give same result", {
  set.seed(123)
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
  update_H_cpp(X,M,W,beta,H,y,delta,alpha,FALSE)

  expect_equal(fit1,H)
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
  update_H_cpp(X,M,W,beta,H,y,delta,alpha,TRUE)

  expect_equal(fit1,H)
})

test_that("update_W R and cpp give same result column normalization", {
  set.seed(123)
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
  update_W_cpp(X,M,H,W,beta,y,delta,alpha,FALSE,2)

  expect_equal(fit1,W)
})

test_that("update_W R and cpp give same result row normalization", {
  set.seed(123)
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
  update_W_cpp(X,M,H,W,beta,y,delta,alpha,FALSE,1)

  expect_equal(fit1,W)
})

test_that("calc_loss gives same results in R and cpp", {
  set.seed(123)
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

test_that("my cdfit_cox_dh gives same result as ncvsurv", {
  set.seed(123)
  n=100
  k=6
  
  X=matrix(runif(n*k,0,5),nrow=n,ncol=k)
  y=runif(n,0,5)
  delta=rbinom(n,1,.5)
  n=nrow(X)
  p=ncol(X)
  y=Surv(y,delta)
  penalty='lasso'
  gamma=switch(penalty, SCAD=3.7, 3)
  alpha=1
  lambda=seq(0,.5,length.out=20)
  eps=1e-4
  max.iter=10000
  convex=TRUE
  dfmax=p
  penalty.factor=rep(1, ncol(X))
  warn=TRUE
  
  
  ## Set up XX, yy, lambda
  tOrder <- order(y[, 1])
  yy <- as.double(y[tOrder, 1])
  Delta <- y[tOrder, 2]
  n <- length(yy)
  XX <- std(X[tOrder, , drop=FALSE])
  if (sys.nframe() > 1 && sys.call(-1)[[1]]=="local_mfdr") return(list(X=XX, time=yy, fail=Delta))
  ns <- attr(XX, "nonsingular")
  penalty.factor <- penalty.factor[ns]
  p <- ncol(XX)
  nlambda <- length(lambda)
  lambda <- sort(lambda, decreasing=TRUE)
  
  
  res2 = cdfit_cox_dh(XX, Delta, penalty, lambda, eps, as.integer(max.iter), as.double(gamma), penalty.factor,
                      alpha, as.integer(dfmax), TRUE, as.integer(warn))
  
  b <- res2
  
  ## Unstandardize
  beta <- matrix(0, nrow=ncol(X), ncol=length(lambda))
  bb <- b/attr(XX, "scale")[ns]
  beta[ns,] <- bb
  
  ## Names
  lam_names <- function(l) {
    if (length(l) > 1) {
      d <- ceiling(-log10(-max(diff(l))))
      d <- min(max(d,4), 10)
    } else {
      d <- 4
    }
    formatC(l, format="f", digits=d)
  }
  varnames <- if (is.null(colnames(X))) paste("V", 1:ncol(X), sep="") else colnames(X)
  dimnames(beta) <- list(varnames, lam_names(lambda))
  
  
  ### ncvsurv
  beta2 = ncvsurv(X=X,y=y,penalty = penalty, lambda=lambda)$beta
  
  
  expect_equal(beta,beta2)
})


test_that("converting cdfit_cox_dh to only accept one lambda gives same result", {
  set.seed(123)
  n=100
  k=6
  
  X=matrix(runif(n*k,0,5),nrow=n,ncol=k)
  y=runif(n,0,5)
  delta=rbinom(n,1,.5)
  n=nrow(X)
  p=ncol(X)
  y=Surv(y,delta)
  penalty='lasso'
  gamma=switch(penalty, SCAD=3.7, 3)
  alpha=1
  lambda=seq(0,.5,length.out=20)
  eps=1e-4
  max.iter=10000
  convex=TRUE
  dfmax=p
  penalty.factor=rep(1, ncol(X))
  warn=TRUE
  
  
  ## Set up XX, yy, lambda
  tOrder <- order(y[, 1])
  yy <- as.double(y[tOrder, 1])
  Delta <- y[tOrder, 2]
  n <- length(yy)
  XX <- std(X[tOrder, , drop=FALSE])
  if (sys.nframe() > 1 && sys.call(-1)[[1]]=="local_mfdr") return(list(X=XX, time=yy, fail=Delta))
  ns <- attr(XX, "nonsingular")
  penalty.factor <- penalty.factor[ns]
  p <- ncol(XX)
  nlambda <- length(lambda)
  lambda <- sort(lambda, decreasing=TRUE)
  
  
  res2 = cdfit_cox_dh(XX, Delta, penalty, lambda, eps, as.integer(max.iter), as.double(gamma), penalty.factor,
                      alpha, as.integer(dfmax), TRUE, as.integer(warn))
  
  b <- res2
  
  ## Unstandardize
  beta <- matrix(0, nrow=ncol(X), ncol=length(lambda))
  bb <- b/attr(XX, "scale")[ns]
  beta[ns,] <- bb
  
  ## Names
  lam_names <- function(l) {
    if (length(l) > 1) {
      d <- ceiling(-log10(-max(diff(l))))
      d <- min(max(d,4), 10)
    } else {
      d <- 4
    }
    formatC(l, format="f", digits=d)
  }
  varnames <- if (is.null(colnames(X))) paste("V", 1:ncol(X), sep="") else colnames(X)
  dimnames(beta) <- list(varnames, lam_names(lambda))
  
  
  ### one lambda
  lambda=lambda[18]
  res3 = cdfit_cox_dh_one_lambda(XX, Delta, penalty, lambda, eps, max.iter,
                                 penalty.factor, alpha)
  
  
  b <- res3
  
  ## Unstandardize
  beta2 <- matrix(0, nrow=ncol(X), ncol=length(lambda))
  bb <- b/attr(XX, "scale")[ns]
  beta2[ns,] <- bb
  
  ## Names
  lam_names <- function(l) {
    if (length(l) > 1) {
      d <- ceiling(-log10(-max(diff(l))))
      d <- min(max(d,4), 10)
    } else {
      d <- 4
    }
    formatC(l, format="f", digits=d)
  }
  varnames <- if (is.null(colnames(X))) paste("V", 1:ncol(X), sep="") else colnames(X)
  dimnames(beta2) <- list(varnames, lam_names(lambda))
  
  
  expect_true(all(abs(beta[,18]-beta2)<.001))
})

test_that("cdfit_cox_dh_one_lambda_it with a=0 gives same result as cdfit_cox_dh_one_lambda 
          with one iteration", {
  set.seed(123)
  n=100
  k=6
  
  X=matrix(runif(n*k,0,5),nrow=n,ncol=k)
  y=runif(n,0,5)
  delta=rbinom(n,1,.5)
  n=nrow(X)
  p=ncol(X)
  y=Surv(y,delta)
  penalty='lasso'
  gamma=switch(penalty, SCAD=3.7, 3)
  alpha=1
  lambda=.05
  eps=1e-4
  max.iter=10000
  convex=TRUE
  dfmax=p
  penalty.factor=rep(1, ncol(X))
  warn=TRUE
  
  
  ## Set up XX, yy, lambda
  tOrder <- order(y[, 1])
  yy <- as.double(y[tOrder, 1])
  Delta <- y[tOrder, 2]
  n <- length(yy)
  XX <- std(X[tOrder, , drop=FALSE])
  if (sys.nframe() > 1 && sys.call(-1)[[1]]=="local_mfdr") return(list(X=XX, time=yy, fail=Delta))
  ns <- attr(XX, "nonsingular")
  penalty.factor <- penalty.factor[ns]
  p <- ncol(XX)
  
  
  res2 = cdfit_cox_dh_one_lambda(XX, Delta, penalty, lambda, eps, 1, 
                                 penalty.factor, alpha)
  res3 = cdfit_cox_dh_one_lambda_it(XX, Delta, penalty, lambda, rep(0,p),
                                    penalty.factor, alpha)
  
  
  
  
  expect_equal(res2,res3)
})


test_that("update_beta_cpp gives same results as manual calculation in R", {
  set.seed(123)
  n=100
  k=6
  
  X=matrix(runif(n*k,0,5),nrow=n,ncol=k)
  y=runif(n,0,5)
  delta=rbinom(n,1,.5)
  n=nrow(X)
  p=ncol(X)
  y=Surv(y,delta)
  penalty='lasso'
  gamma=switch(penalty, SCAD=3.7, 3)
  alpha=1
  lambda=.05
  eps=1e-4
  max.iter=10000
  convex=TRUE
  dfmax=p
  penalty.factor=rep(1, ncol(X))
  warn=TRUE
  
  
  ## Set up XX, yy, lambda
  tOrder <- order(y[, 1])
  yy <- as.double(y[tOrder, 1])
  Delta <- y[tOrder, 2]
  n <- length(yy)
  XX <- std(X[tOrder, , drop=FALSE])
  if (sys.nframe() > 1 && sys.call(-1)[[1]]=="local_mfdr") return(list(X=XX, time=yy, fail=Delta))
  ns <- attr(XX, "nonsingular")
  penalty.factor <- penalty.factor[ns]
  p <- ncol(XX)
  
  
  res2 = cdfit_cox_dh_one_lambda_it(XX, Delta, penalty, lambda, rep(0,p),
                                    penalty.factor, alpha)
  
  b <- res2
  
  ## Unstandardize
  beta <- matrix(0, nrow=ncol(X), ncol=length(lambda))
  bb <- b/attr(XX, "scale")[ns]
  beta[ns,] <- bb
  
  beta2 = update_beta_cpp(X,y,penalty,alpha,lambda,rep(0,p))
  
  
  expect_equal(beta,beta2)
})

test_that("optimize_loss_cpp converges", {
  set.seed(123)
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
  lambda=.1
  eta=1
  tol=1e-5
  maxit=10000
  verbose=FALSE
  WtX=FALSE
  norm_type=2
  penalty='lasso'
  
  expect_no_warning(optimize_loss_cpp(X,M,H,W,beta,y,delta,alpha,lambda,eta,tol,maxit,
                          verbose,WtX,norm_type,penalty,FALSE))
})


test_that("run_coxNMF runs without warning", {
  set.seed(123)
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
  lambda=.05
  eta=1
  tol=1e-5
  maxit=10000
  verbose=FALSE
  WtX=FALSE
  norm_type=2
  penalty='lasso'

  expect_no_warning(run_coxNMF(X,y,delta,k,alpha,lambda,eta))
})
