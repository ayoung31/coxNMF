#' @export
recommend_alpha <- function(X,y,delta,k,nalpha,maxit=15){
  if(nalpha < 1){
    warning('nalpha must be a positive integer')
  }
  p <- nrow(X)
  n <- ncol(X)
  H0 <- matrix(runif(n*k,0,1),nrow=k)
  W0 <- matrix(runif(p*k,0,1),nrow=p)
  beta0 <- rep(0,k)
  fit0 <- optimize_loss(X,H0,W0,beta0,y,delta,alpha=0,maxit=maxit)
  if(fit0$nmf_loss < fit0$surv_loss){
    alpha5050 <- fit0$nmf_loss / fit0$surv_loss
  }else{
    alpha5050 <- fit0$surv_loss / fit0$nmf_loss
  }
  if(nalpha %% 2 > 0){
    nalpha=nalpha+1
  }
  grid <- 2:(nalpha/2 + 1)
  alpha_grid <- c(alpha5050*grid,alpha5050,alpha5050/grid)
  
  return(list(alpha5050=alpha5050, alpha_grid=alpha_grid))
}