#' @export
init = function(X, M, y, delta, k, alpha, lambda, eta, lambdaW, lambdaH,
                tol = 1e-6, imaxit = 30, verbose = TRUE, ninit = 20){

  p = nrow(X)
  n = ncol(X)
  loss_best = Inf
  for(i in 1:ninit){
    set.seed(i)
    H0 = matrix(runif(n*k,0,max(X)),nrow=k)
    W0 = matrix(runif(p*k,0,max(X)),nrow=p)
    #beta0 = runif(k,-1,1)
    beta0 = rep(0,k)

    fit = optimize_loss_cpp(X, M, y, delta, W0, H0, beta0, 
                            alpha, lambda, eta,
                            lambdaW, lambdaH,
                            tol, imaxit, verbose, TRUE)
    
    loss = abs(fit$loss$loss)
    if(loss < loss_best){
      loss_best=loss
      fit_best=fit
      selected=i
    }
    print(i)
  }
  print(selected)
  return(list(W0=fit_best$W,H0=fit_best$H,beta0=fit_best$beta))
}
