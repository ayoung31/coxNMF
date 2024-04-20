#' @export
init <- function(X,M,y,delta,k,alpha,lambda=NULL,eta=NULL,ninit=5,maxit=10,warmup=1){
  
  if(warmup >= maxit){
    warning('Maxit must be larger than warmup')
  }
  p <- nrow(X)
  n <- ncol(X)
  loss_best <- 1/0
  for(i in 1:ninit){
    H0 <- matrix(runif(n*k,0,1),nrow=k)
    W0 <- matrix(runif(p*k,0,1),nrow=p)
    #beta0 <- runif(k,-1,1)
    beta0 <- rep(0,k)
    fit0 <- optimize_loss(X=X,M=M,H0=H0,W0=W0,beta0=beta0,y=y,delta=delta,
                          alpha=0,lambda=0,eta=0,maxit=warmup)
    fit <- optimize_loss(X=X,M=M,H0=fit0$H,W0=fit0$W,beta0=fit0$beta,y=y,delta=delta,
                         alpha=alpha,lambda=lambda,eta=eta,maxit=max(maxit-warmup,0))
    loss <- fit$loss
    if(loss < loss_best){
      loss_best=loss
      fit_best=fit
    }
  }
  return(list(W0=fit_best$W,H0=fit_best$H,beta0=fit_best$beta))
}
