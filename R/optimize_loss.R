#' @export
optimize_loss <- function(X,H0=NULL,W0,beta0,y,delta,alpha,lambda=0,eta=1,tol=1e-4,maxit=1000){
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
    W <- update_W(X,H,W)
    
    # Update H
    H <- update_H(X,W,beta,H,y,delta,alpha)
    
    # Update beta
    beta <- update_beta(H,y,delta,lambda,eta)
    
    # Normalization
    S <- rowSums(H)
    Sinv <- diag(1/S)
    S <- diag(S)
    W <- W%*%S
    H <- Sinv%*%H
    beta <- S%*%matrix(beta,ncol=1)
    
    
    # Calculate loss
    l <- calc_loss(X,W,H,beta,alpha,y,delta,theta,lambda)
    loss <- l$loss
    
    eps <- abs(loss - loss_prev)/loss_prev
    
    print(sprintf("iter: %d eps: %.4f loss: %.1f",it,eps,loss))
    
    it <- it + 1
    if(it==maxit){
      warning("Iteration limit reached without convergence")
    }
    
    print(sprintf("iter: %d eps: %.4f loss: %.1f",it,eps,loss))
    
  }
  
  # refit beta with standardized H
  #nb <- update_beta(H,y,delta,theta,lambda,eta,stdize=FALSE)
  
  return(list(H=H,W=W,beta=beta,H0=H0,W0=W0,beta0=beta0,X=X,loss=loss,eps=eps,surv_loss=l$surv_loss,nmf_loss=l$nmf_loss))
}