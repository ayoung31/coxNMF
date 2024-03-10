#' @export
init <- function(X,y,delta,k,alpha,ninit=10,maxit=15){
  p <- nrow(X)
  n <- ncol(X)
  loss_best <- 1/0
  for(i in 1:ninit){
    H0 <- matrix(runif(n*k,0,1),nrow=k)
    W0 <- matrix(runif(p*k,0,1),nrow=p)
    #beta0 <- runif(k,-1,1)
    beta0 <- rep(0,k)
    fit <- optimize_loss(X,H0,W0,beta0,y,delta,alpha=alpha,maxit=maxit)
    loss <- fit$loss
    if(loss < loss_best){
      loss_best=loss
      fit_best=fit
    }
  }
  return(list(W0=fit_best$W,H0=fit_best$H,beta0=fit_best$beta))
}