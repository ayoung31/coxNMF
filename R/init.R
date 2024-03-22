#' @export
init <- function(X,y,delta,k,alpha,ninit=10,maxit=15){
  
  if(maxit <10){
    warning('Maxit must be at least 10')
  }
  p <- nrow(X)
  n <- ncol(X)
  loss_best <- 1/0
  for(i in 1:ninit){
    H0 <- matrix(runif(n*k,0,1),nrow=k)
    W0 <- matrix(runif(p*k,0,1),nrow=p)
    #beta0 <- runif(k,-1,1)
    beta0 <- rep(0,k)
    fit0 <- optimize_loss(X,H0,W0,beta0,y,delta,alpha=0,maxit=10)
    fit <- optimize_loss(X,fit0$H,fit0$W,fit0$beta,y,delta,alpha=alpha,maxit=max(maxit-10,0))
    loss <- fit$loss
    if(loss < loss_best){
      loss_best=loss
      fit_best=fit
    }
  }
  return(list(W0=fit_best$W,H0=fit_best$H,beta0=fit_best$beta))
}