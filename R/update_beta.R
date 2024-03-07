#' @export
update_beta <- function(H,y,delta,lambda=0,eta=1){
  
  y_surv <- survival::Surv(y,delta)
  fit <- glmnet::glmnet(t(H),y_surv,family='cox',
                        lambda=lambda,alpha=eta)
  
  return(as.numeric(coef(fit)))
}