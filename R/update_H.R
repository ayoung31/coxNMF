#' @export
update_H <- function(X,W,beta,H,y,delta,alpha){
  N <- ncol(H)
  Hnew <- H
  l <- H
  beta <- matrix(beta,ncol=1)
  for(r in 1:N){
    l[,r] <- delta[r]*beta
    for(s in 1:N){
      l[,r] <- l[,r] - as.numeric((delta[s]*(y[r]>=y[s])*
                          exp(t(beta)%*%H[,r])/
                          sum(((y>=y[s])*t(exp(t(beta)%*%H)))))) * beta
    }
  }
  Hnew <- (H / (t(W)%*%W%*%H)) * ((t(W)%*%X) + (alpha/2)*pmax(l,0))
  return(Hnew)
}