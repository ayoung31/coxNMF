#' @export
update_beta <- function(H,y,delta,lambda,eta){
  y_surv <- survival::Surv(y,delta)
  
  fit=glmnet::glmnet(t(H),y_surv,family='cox',alpha = eta,lambda=lambda)
  
  return(as.numeric(coef(fit)))
}
