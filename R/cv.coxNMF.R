#' @export
cv.coxNMF <- function(dat,nfold,perc_miss,k,alpha,lambda=NULL,eta=NULL,seed){
  
  folds <- get_folds(n=ncol(dat$Train$X),nfold=nfold,seed=seed)
  
  fits <- list()
  Htest <- list()
  ctest <- numeric()
  ctrain <- numeric()
  rtrain <- numeric()
  rtest <- numeric()
  Trains <- list()
  Tests <- list()
  metric <- numeric()
  for(i in 1:nfold){
    Train <- list()
    Test <- list()
    Train$X <- dat$Train$X[,-folds[[i]]]
    Train$s <- dat$Train$s[-folds[[i]],]
    Test$X <- dat$Train$X[,folds[[i]]]
    Test$s <- dat$Train$s[folds[[i]],]
    
    #within Train$X sample ~30% of cells to be missing
    Train$M <- get_mask(Train,perc_miss=perc_miss)
    
    Trains[[i]] <- Train
    Tests[[i]] <- Test
    
    fit = run_coxNMF(dat = Train, k = k, alpha = alpha, lambda = lambda,eta = eta)
    print(i)
    fits[[i]] <- fit
    
    rtrain[i] <- fit$fit$nmf_loss
    if(sum(1-Train$M)!=0){
      rtest[i] <- norm((1-Train$M)*(fit$fit$X-fit$fit$W%*%fit$fit$H),'F')^2 / sum(1-Train$M)
    }else{
      rtest[i] <- NA
    }
    
    
    Htest[[i]] = NMF::.fcnnls(fit$fit$W,Test$X)$coef
    ctest[i] = cvwrapr::getCindex(t(Htest[[i]])%*%fit$fit$beta,Test$s)
    ctrain[i] = fit$surv$concordance[6]
    
    # add testing deviance here
    
    metric[i] = ctest[i]/rtest[i]
  }
  return(list(fits=fits,Htest=Htest,ctest=ctest,ctrain=ctrain,rtest=rtest,
              rtrain=rtrain,metric=metric,Trains=Trains,Tests=Tests,folds=folds))
}