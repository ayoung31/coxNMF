update_W <- function(X,H,W){
  Wnew <- W * (X%*%t(H)) / (W%*%H%*%t(H))
  return(Wnew)
}