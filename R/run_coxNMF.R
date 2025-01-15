#' Runs the coxNMF model
#' 
#' This function runs the block updating scheme for the coxNMF model for specified parameters. If initialization is not provided, the function first calls the init function.
#' 
#' @param X matrix of gene expression values of dimension p genes by n samples
#' @param y a vector of n survival times 
#' @param delta a vector of n event indicators 0=censored, 1=died
#' @param k the rank of the NMF decomposition
#' @param alpha the balancing parameter for NMF and survival contribution to loss
#' @param lambda parameter for penalty on the coefficients in survival model
#' @param eta parameter for balance of l1 and l2 penalty on the coefficients, eta=1 is all l1
#' @param gamma parameter for l1 penalty on W
#' @param H0 matrix of initial values for H of dimension k x n
#' @param W0 matrix of initial values for W of dimension p x k
#' @param beta0 vector of k initial values for beta, the survival model coefficients
#' @param M matrix of 0s and 1s representing the masking matrix of dimension p x n
#' @param WtX boolean indicator if TRUE, rows of WtX are the covariates fed to the survival model, if FALSE, rows of H are the covariates
#' @param verbose if TRUE prints loss at each iteration
#' @param tol convergence criteria
#' @param maxit maximum number of iterations
#' @param penalty the type of penalty on the survival coefficients
#' @param BFGS whether to run BFGS for updating W
#' @param ninit number of initializations, ignored if initializations provided
#' @param imaxit max number of iterations for each initialization, ignored if initializations provided
#'
#' @return a list containing the fitted W, H, and beta values, as well as the values of the loss
#' 
#' 
#' @export
run_coxNMF = function(X, y, delta, k, alpha, lambda, eta, gamma=0, H0 = NULL, W0 = NULL, 
                      beta0 = NULL, M = NULL, WtX = TRUE,
                      verbose = FALSE, tol = 1e-6, maxit = 500,
                      penalty = 'lasso', BFGS=TRUE, ninit=10, imaxit=15, ...){
  # Add all error checking here. This will be the primary function
  
  if(is.null(M)){
    M = matrix(1,nrow=nrow(X),ncol=ncol(X))
  }
  
  # Initialize
  if(is.null(H0) | is.null(W0) | is.null(beta0)){
    print("initializing ...")
    fit0 = init(X=X, M=M, y=y, delta=delta, k=k, alpha=alpha, lambda=lambda, 
                eta=eta, gamma=gamma, WtX=WtX, BFGS=BFGS, penalty=penalty, 
                verbose=verbose, tol=tol, ninit=ninit, imaxit=imaxit, ...)
    H0 = fit0$H0
    W0 = fit0$W0
    beta0 = fit0$beta0
  }
  
  
  # Run the model
  fit = optimize_loss_cpp(X, M, H0, W0, beta0, y, delta, alpha, 
                          lambda, eta, tol, maxit, verbose, WtX, 
                          penalty, BFGS, gamma)
  
  return(fit)
}
