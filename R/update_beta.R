#' @export
update_beta <- function(H,W,X,M,y,delta,lambda,eta,WtX){
  y_surv <- survival::Surv(y,delta)
  
  if(WtX){
    pred = t(t(W)%*%(M*X))
  }else{
    pred = t(H)
  }
  
  fit=glmnet::glmnet(pred,y_surv,family='cox',alpha = eta,lambda=lambda)
  
  return(as.numeric(coef(fit)))
}
