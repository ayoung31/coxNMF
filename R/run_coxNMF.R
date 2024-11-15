#' @export
run_coxNMF = function(X, y, delta, k, alpha, lambda, eta, H0 = NULL, W0 = NULL, 
                      beta0 = NULL, M = NULL, WtX = FALSE,
                      verbose = FALSE, norm_type = 2, tol = 1e-6, maxit = 10000,
                      penalty = 'lasso',step=1e-2,mo=.8,...){
  # Add all error checking here. This will be the primary function
  
  if(is.null(M)){
    M = matrix(1,nrow=nrow(X),ncol=ncol(X))
  }
  
  # Initialize
  if(is.null(H0) | is.null(W0) | is.null(beta0)){
    print("initializing ...")
    fit0 = init(X, M, y, delta, k, alpha, lambda, eta, WtX, norm_type, 
                penalty, verbose, tol,...)
    H0 = fit0$H0
    W0 = fit0$W0
    beta0 = fit0$beta0
  }
  
  
  # Run the model
  fit = optimize_loss_cpp(X, M, H0, W0, beta0, y, delta, alpha, 
                          lambda, eta, tol, maxit, verbose, WtX, norm_type, 
                          penalty, FALSE, step, mo)
  
  return(fit)
}
