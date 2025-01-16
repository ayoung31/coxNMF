#' @export
recommend_alpha <- function(X,M,y,delta,k,nalpha,maxit=15,WtX=TRUE,eta=0,lambda=0, norm.type){
  if(nalpha < 1){
    warning('nalpha must be a positive integer')
  }
  p <- nrow(X)
  n <- ncol(X)
  set.seed(12)
  H0 = matrix(runif(n*k,0,max(X)),nrow=k)
  W0 = matrix(runif(p*k,0,max(X)),nrow=p)
  #W0 = W0 %*% diag(1/colSums(W0))
  beta0 = rep(0,k)#runif(k,-1,1)
  fit0 <- optimize_loss_cpp(X,M,H0,W0,beta0,y,delta,0,lambda,eta,1e-6,maxit,
                            TRUE,WtX,'lasso',TRUE,0)
  
  alpha5050 = fit0$loss$nmf_loss/(fit0$loss$nmf_loss-fit0$loss$surv_loss*sum(M))
  maxalpha=alpha5050/10
  return(maxalpha)
}
