#' @export
update_beta <- function(H,y,delta,lambda,eta){
  y_surv <- survival::Surv(y,delta)
  
  if(is.null(eta)){
    eta = seq(0,1,by=.1)
  }
  
  if(length(lambda)<2 & !is.null(lambda)){
    fit=glmnet::glmnet(t(H),y_surv,family='cox',alpha = eta,lambda=lambda)
    beta = coef(fit,s=lambda)
    loglike=((1-fit$dev.ratio)*fit$nulldev)/-2
    lam_best=lambda
    eta_best=eta
  }else{
    cvm_best=99999
    for(e in eta){
      # note if lambda is null, cv.glmnet will supply lambda grid
      fit <- glmnet::cv.glmnet(t(H),y_surv,family='cox',alpha = e,lambda=lambda)
      # get 1se cvm
      cvm = fit$cvm[which(fit$lambda==fit$lambda.min)]
      if(cvm < cvm_best){
        cvm_best=cvm
        fit_best=fit
        eta_best=e
      }
    }
    beta = coef(fit_best,s=fit_best$lambda.min)
    lam_best = fit_best$lambda.min
    dev.ratio=fit_best$glmnet.fit$dev.ratio[fit_best$lambda==lam_best]
    dev = (1-dev.ratio)*fit_best$glmnet.fit$nulldev
    loglike = dev / -2
  }
  
  
  return(list(beta=as.matrix(beta),loglike=loglike,lambda=lam_best,eta=eta_best)) # not going to use lambda and eta in future iterations, so no need to return
}
