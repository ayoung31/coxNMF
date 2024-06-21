#' @export
run_coxNMF = function(X, y, delta, k, alpha, lambda, eta, M = NULL, WtX = FALSE,
                      verbose = FALSE, norm_type = 2, tol = 1e-6, maxit = 10000,
                      penalty = 'lasso',...){
  # Add all error checking here. This will be the primary function
  
  if(is.null(M)){
    M = matrix(1,nrow=nrow(X),ncol=ncol(X))
  }
  
  # Initialize
  fit0 = init(X, M, y, delta, k, alpha, lambda, eta, WtX, norm_type, 
              penalty, verbose, tol,...)
  
  # Run the model
  fit = optimize_loss_cpp(X, M, fit0$H0, fit0$W0, fit0$beta0, y, delta, alpha, 
                          lambda, eta, tol, maxit, verbose, WtX, norm_type, penalty, FALSE)
  
  return(fit)
}
