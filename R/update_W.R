#' @export
update_W <- function(X,M,H,W){
  Wnew <- W * ((M*X)%*%t(H)) / ((M*(W%*%H))%*%t(H))
  return(Wnew)
}