#' @export
run_coxNMF = function(dat, k, alpha, lambda, eta, WtX,...){
  # now do survival
  fit0 = init(X = dat$X,M=dat$M,y = dat$s[,1],delta = dat$s[,2],k = k,alpha = alpha, lambda=lambda, eta=eta, WtX=WtX, ...) # new init adjustment here
  fit = optimize_loss(X = dat$X, dat$M, y = dat$s[,1],delta = dat$s[,2],H0=fit0$H0,
                      W0 = fit0$W0,beta0=fit0$beta0, alpha = alpha,lambda = lambda, 
                      eta = eta, WtX=WtX, ...)
  
  return(fit)
}
