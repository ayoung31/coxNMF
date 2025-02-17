#' @export
run_coxNMF = function(X, y, delta, k, alpha, M = NULL, W0 = NULL, H0 = NULL,
                      beta0 = NULL, lambda = 0, eta = 0, 
                      tol = 1e-6, maxit = 3000, verbose = TRUE, 
                      ninit = 30, imaxit = 100){
  # Add all error checking here. This will be the primary function
  
  if(is.null(M)){
    M = matrix(1,nrow=nrow(X),ncol=ncol(X))
  }
  
  # Initialize
  if(is.null(H0) | is.null(W0) | is.null(beta0)){
    print("initializing ...")
    fit0 = init(X=X, M=M, y=y, delta=delta, k=k, 
                alpha=alpha, lambda=lambda, eta=eta,
                verbose=verbose, tol=tol, imaxit=imaxit, ninit=ninit)
    H0 = fit0$H0
    W0 = fit0$W0
    beta0 = fit0$beta0
  }
  
  # Run the model
  fit = optimize_loss_cpp(X, M, y, delta, W0, H0, beta0, 
                          alpha, lambda, eta, 
                          tol, maxit, verbose, FALSE)
  
  return(fit)
}
