#' @export
update_W <- function(X,M,H,W,beta,y,delta,alpha,WtX,norm.type){
  if(WtX){
    N <- ncol(H)
    Hnew <- H
    l <- H
    beta <- matrix(beta,ncol=1)
    
    a1 <- exp(t(beta)%*%t(W)%*%(M*X))
    a2 <- sweep(M*X,2,a1,'*')
    
    ind = matrix(NA,nrow=N,ncol=N)
    for(i in 1:N){
      ind[,i] = (y>=y[i])^2
    }
    
    a3 <- sweep(a2 %*% ind,2,a1 %*% ind,'/')
    l <- kronecker((M*X-a3)%*%delta,t(beta))
    Wnew <- (W / ((M*(W%*%H))%*%t(H))) * ((M*X)%*%t(H) + (alpha/2)*pmax(l,0))
    
  }else{
    Wnew <- W * ((M*X)%*%t(H)) / ((M*(W%*%H))%*%t(H))
  }
  
  if(norm.type==1){
    Wnew <- diag(1/rowSums(Wnew))%*%Wnew
  }else if(norm.type==2){
    Wnew <- Wnew%*%diag(1/colSums(Wnew))
  }
  
  return(Wnew)
}