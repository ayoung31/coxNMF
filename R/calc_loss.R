#' @export
calc_loss <- function(X,M,W,H,beta,alpha,y,delta,lambda,eta,WtX){
  N <- ncol(H)
  beta <- matrix(beta,ncol=1)
  nmf_loss <- (norm(M*(X-W%*%H),'F')^2)/sum(M)
  
  if(WtX){
    a1 <- t(beta)%*%t(W)%*%(M*X)
  }else{
    a1 <- t(beta)%*%H
  }
  
  ind = matrix(NA,nrow=N,ncol=N)
  for(i in 1:N){
    ind[,i] = (y>=y[i])^2
  }
  surv_loss = 2*sum(delta*(a1 - log(exp(a1)%*%ind)))/n

  penalty <- lambda*((1-eta)*sum(beta^2)/2 + eta*sum(abs(beta)))
  loss <- nmf_loss - alpha*(surv_loss - penalty)
  return(list(loss=loss,nmf_loss=nmf_loss,surv_loss=surv_loss))
}
