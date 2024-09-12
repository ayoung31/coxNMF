#' @export
optimize_loss = function(X,M,H0,W0,beta0,y,delta,alpha,lambda,eta,
                          tol=1e-6,maxit=10000,verbose=FALSE,WtX,norm.type){
  #initialize
  H = H0
  #W = W0
  k=nrow(W0)
  beta = beta0
  loss = 1e-8
  # loss = calc_loss(X=X,M=M,W=W,H=H,beta=beta,alpha=alpha,y=y,delta=delta,
  #                   lambda=lambda,eta=eta,WtX=WtX)$loss
  eps = 1
  it = 0
  s = matrix(c(y,delta),byrow = TRUE,nrow=length(y))
  
  while(eps > tol & it <= maxit){
    loss_prev = loss
    # Update W
    update_W_cpp(X,M,H,W0,beta,y,delta,alpha,WtX,2)
    
    # Update H
    # update_H_(X=X,M=M,W=W,beta=beta,H=H,y=y,delta=delta,alpha=alpha,WtX=WtX)
    # call optimx
    
    # Update beta
    beta = update_beta_cpp(t(H),s,'lasso',alpha,lambda,rep(0,k))

    # Calculate loss
    l = calc_loss_cpp(X,M,W,H,beta,alpha,y,delta,lambda,eta,WtX)
    loss = as.numeric(l[1])
    eps = abs(loss - loss_prev)/loss_prev
    
    
    it = it + 1
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
