#' @export
optimize_loss_cpp <- function(X,M,H0,W0,beta0,y,delta,alpha,lambda,eta,
                          tol=1e-6,maxit=10000,verbose=FALSE,WtX,norm.type){
  #initialize
  H <- H0
  W <- W0
  beta <- beta0
  loss <- calc_loss_cpp(X=X,M=M,W=W,H=H,beta=beta,alpha=alpha,y=y,delta=delta,
                        lambda=lambda,eta=eta,WtX=WtX)$loss
  
  eps <- 1
  it <- 0
  
  while(eps > tol & it <= maxit){
    loss_prev <- loss
    eps_prev <- eps
    
    # Update W
    W <- update_W_cpp(X=X,M=M,H=H,W=W,beta=beta,y=y,delta=delta,alpha=alpha,WtX=WtX,norm_type=norm.type)
    
    # Update H
    H <- update_H_cpp(X=X,M=M,W=W,beta=beta,H=H,y=y,delta=delta,alpha=alpha,WtX=WtX)
    
    # Update beta
    beta <- update_beta(H=H,W=W,X=X,M=M,y=y,delta=delta,lambda=lambda,eta=eta,WtX=WtX)
    
    # Calculate loss
    l <- calc_loss_cpp(X=X,M=M,W=W,H=H,beta=beta,alpha=alpha,y=y,delta=delta,lambda=lambda,eta=eta,WtX=WtX)
    loss <- l$loss
    
    eps <- abs(loss - loss_prev)/loss_prev
    
    
    it <- it + 1
    if(verbose){
      print(sprintf("iter: %d eps: %.8f loss: %.8f",it,eps,loss))
    }
    
    if(it==maxit){
      warning("Iteration limit reached without convergence")
    }
    
  }
  
  # Define return objects
  fit = list(W=W,H=H,beta=beta)
  data = list(X=X,y=y,delta=delta,M=M)
  init = list(W0=W0,H0=H0,beta0=beta0)
  params = list(alpha=alpha,lambda=lambda,eta=eta)
  options = list(tol=tol,maxit=maxit,verbose=verbose,
                 WtX=WtX,norm.type=norm.type)
  
  return(list(fit=fit,data=data,init=init,params=params,options=options,loss=l,niter=it))
}
