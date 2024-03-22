#' @export
calc_loss <- function(X,W,H,beta,alpha,y,delta,lambda,eta){
  N <- ncol(H)
  P <- ncol(W)
  beta <- matrix(beta,ncol=1)
  nmf_loss <- (norm(X-W%*%H,'F')^2)/(N*P)
  surv_loss <- 0
  for(i in 1:N){
    surv_loss <- surv_loss + 
      delta[i]*(t(beta)%*%H[,i] - 
                  log(sum((y>=y[i])*t(exp(t(beta)%*%H)))))
  }
  surv_loss <- surv_loss/N
  penalty <- (lambda/2)*((1-eta)*sum(beta^2) + eta*sum(abs(beta)))
  loss <- nmf_loss - alpha*(surv_loss + penalty)
  return(list(loss=loss,nmf_loss=nmf_loss,surv_loss=surv_loss))
}