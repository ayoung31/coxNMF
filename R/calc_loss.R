#' @export
calc_loss <- function(X,W,H,beta,alpha,y,delta,lambda=1,eta=0){
  N <- ncol(H)
  beta <- matrix(beta,ncol=1)
  nmf_loss <- norm(X-W%*%H,'F')^2
  surv_loss <- 0
  for(i in 1:N){
    surv_loss <- surv_loss + 
      delta[i]*(t(beta)%*%H[,i] - 
                  log(sum((y>=y[i])*t(exp(t(beta)%*%H)))))
  }
  loss <- nmf_loss + alpha*(-2*surv_loss/N + lambda*(((1-eta)/2)*sum(beta^2) + eta*sum(abs(beta))))
  return(list(loss=loss,nmf_loss=nmf_loss,surv_loss=surv_loss))
}