#' Initializes the coxNMF model
#' 
#' Outputs the model with the smallest loss value across many random initializations
#'
#' @param X matrix of gene expression values of dimension p genes by n samples
#' @param y a vector of n survival times 
#' @param delta a vector of n event indicators 0=censored, 1=died
#' @param k the rank of the NMF decomposition
#' @param alpha the balancing parameter for NMF and survival contribution to loss
#' @param lambda parameter for penalty on the coefficients in survival model
#' @param eta parameter for balance of l1 and l2 penalty on the coefficients, eta=1 is all l1
#' @param gamma parameter for l1 penalty on W
#' @param WtX boolean indicator if TRUE, rows of WtX are the covariates fed to the survival model, if FALSE, rows of H are the covariates
#' @param BFGS whether to run BFGS for updating W
#' @param penalty the type of penalty on the survival coefficients
#' @param verbose if TRUE prints loss at each iteration
#' @param tol convergence criteria
#' @param ninit number of initializations
#' @param imaxit max number of iterations for each initialization
#'
#' @export
init = function(X, M, y, delta, k, alpha, lambda=0, eta=0, gamma=0, WtX=TRUE, BFGS=TRUE,
                 penalty='lasso', verbose=FALSE, tol=1e-6, ninit=10, imaxit=15){
  
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
                              tol, imaxit, verbose, WtX, penalty,BFGS,gamma)
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
