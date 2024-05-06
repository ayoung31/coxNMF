#' @export
init <- function(X,M,y,delta,k,alpha,WtX,lambda,eta,ninit=5,imaxit=10,warmup=1,verbose=verbose){
  
  if(warmup >= imaxit){
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
                          alpha=0,lambda=0,eta=0,maxit=warmup,WtX=WtX,verbose=verbose)
    fit <- optimize_loss(X=X,M=M,H0=fit0$fit$H,W0=fit0$fit$W,beta0=fit0$fit$beta,y=y,delta=delta,
                         alpha=alpha,lambda=lambda,eta=eta,maxit=max(imaxit-warmup,0),WtX=WtX,verbose=verbose)
    loss <- fit$loss$loss
    if(loss < loss_best){
      loss_best=loss
      fit_best=fit
    }
  }
  return(list(W0=fit_best$fit$W,H0=fit_best$fit$H,beta0=fit_best$fit$beta))
}
