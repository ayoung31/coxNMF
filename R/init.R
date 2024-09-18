#' @export
init = function(X, M, y, delta, k, alpha, lambda, eta, WtX, norm.type, 
                 penalty, verbose, tol, ninit=5, imaxit=10, warmup=1){
  
  # if(warmup >= imaxit){
  #   warning('Maxit must be larger than warmup')
  # }
  p = nrow(X)
  n = ncol(X)
  loss_best = 1/0
  for(i in 1:ninit){
    H0 = matrix(runif(n*k,0,max(X)),nrow=k)
    W0 = matrix(runif(p*k,0,max(X)),nrow=p)
    #beta0 = runif(k,-1,1)
    beta0 = rep(0,k)

    fit = optimize_loss_cpp(X, M, H0, W0, beta0, y, delta, alpha, lambda, eta,
                              tol, imaxit, verbose, WtX, norm.type, penalty, TRUE)
    # fit = optimize_loss_cpp(X, M, fit0$H, fit0$W, fit0$beta, y, delta, alpha,
    #                          lambda, eta, tol, max(imaxit-warmup,0), verbose, 
    #                          WtX, norm.type, penalty, TRUE)
    
    loss = fit$loss$loss
    if(loss < loss_best){
      loss_best=loss
      fit_best=fit
    }
    print(i)
  }
  return(list(W0=fit_best$W,H0=fit_best$H,beta0=fit_best$beta))
}
