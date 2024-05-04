#' @export
update_H <- function(X,M,W,beta,H,y,delta,alpha){
  N <- ncol(H)
  beta = matrix(beta,ncol=1)
  a1 = exp(t(beta)%*%H)
  ind = matrix(NA,nrow=N,ncol=N)
  for(i in 1:N){
    ind[,i] = (y>=y[i])^2
  }
  
  a2 = sweep(t(ind),2,a1,'*')
  a3 = as.vector(a1 %*% ind)
  a4 = (1/a3) * a2 # multiply each column of a2 by 1/a3 vector
  
  l=kronecker(t(delta) - (t(delta) %*% a4),beta)
  
  Hnew <- (H / (t(W)%*%(M*(W%*%H)))) * ((t(W)%*%(M*X)) + (alpha/2)*pmax(l,0))
  return(Hnew)
}
