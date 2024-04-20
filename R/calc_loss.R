#' @export
calc_loss <- function(X,M,W,H,beta,alpha,y,delta,lambda,eta,loglike){
  N <- ncol(H)
  #P <- nrow(W)
  beta <- matrix(beta,ncol=1)
  nmf_loss <- (norm(M*(X-W%*%H),'F')^2)/sum(M)
  surv_loss <- loglike/N
  penalty <- (lambda/2)*((1-eta)*sum(beta^2) + eta*sum(abs(beta)))
  loss <- nmf_loss - alpha*(surv_loss - penalty)
  return(list(loss=loss,nmf_loss=nmf_loss,surv_loss=surv_loss))
}
