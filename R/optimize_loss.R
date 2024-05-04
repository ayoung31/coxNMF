#' @export
optimize_loss <- function(X,M,H0=NULL,W0,beta0,y,delta,alpha,lambda=0,eta=0,
                          tol=1e-4,maxit=1000,verbose=FALSE,normalize=TRUE){
  #initialize
  H <- H0
  W <- W0
  beta <- beta0
  loss <- 0
  eps <- 1
  it <- 0
  
  while(eps > tol & it < maxit){
    loss_prev <- loss
    eps_prev <- eps
    
    # Update W
    W <- update_W(X=X,M=M,H=H,W=W)
    # add row normalization here
    W <- diag(1/rowSums(W))%*%W
    
    # Update H
    H <- update_H(X=X,M=M,W=W,beta=beta,H=H,y=y,delta=delta,alpha=alpha)
    
    # Update beta
    beta <- update_beta(H=H,y=y,delta=delta,lambda=lambda,eta=eta)
    
    # Calculate loss
    l <- calc_loss(X=X,M=M,W=W,H=H,beta=beta,alpha=alpha,y=y,delta=delta,lambda=lambda,eta=eta)
    loss <- l$loss
    
    eps <- abs(loss - loss_prev)/loss_prev
    
    if(verbose){
      print(sprintf("iter: %d eps: %.4f loss: %.1f",it,eps,loss))
    }
    
    it <- it + 1
    if(it==maxit){
      warning("Iteration limit reached without convergence")
    }
    
  }
  
  # refit beta with standardized H
  #nb <- update_beta(H,y,delta,theta,lambda,eta,stdize=FALSE)
  
  return(list(H=H,W=W,beta=beta,H0=H0,W0=W0,beta0=beta0,X=X,loss=loss,eps=eps,surv_loss=l$surv_loss,nmf_loss=l$nmf_loss))
  # we may want to return selected lambda and eta here
}
