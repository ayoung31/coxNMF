#' @export
cv.coxNMF <- function(dat,nfold,perc_miss,k,alpha,lambda,eta,seed,WtX,...){
  
  folds <- get_folds(n=ncol(dat$Train$X),nfold=nfold,seed=seed)
  
  fits <- list()
  Htest <- list()
  ctest <- numeric()
  ctrain <- numeric()
  rtrain <- numeric()
  rmask <- numeric()
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
    
    Tests[[i]] <- Test
    
    coxNMF = run_coxNMF(dat = Train, k = k, alpha = alpha, lambda = lambda,eta = eta,WtX=WtX,...)
    print(i)
    fits[[i]] <- coxNMF
    
    rtrain[i] <- coxNMF$fit$nmf_loss
    if(sum(1-Train$M)!=0){
      rmask[i] <- norm((1-Train$M)*(coxNMF$fit$X-coxNMF$fit$W%*%coxNMF$fit$H),'F')^2 / sum(1-Train$M)
    }else{
      rmask[i] <- NA
    }
    
    if(WtX){
      ctrain[i] = cvwrapr::getCindex(t(t(coxNMF$fit$W)%*%Train$X)%*%coxNMF$fit$beta,Train$s)
      Htest[[i]] = t(coxNMF$fit$W)%*%Test$X # Htest is irrelevant for WtX version
      
    }else{
      ctrain[i] = cvwrapr::getCindex(t(coxNMF$fit$H)%*%coxNMF$fit$beta,Train$s)
      Htest[[i]] = NMF::.fcnnls(coxNMF$fit$W,Test$X)$coef
    }
    ctest[i] = cvwrapr::getCindex(t(Htest[[i]])%*%coxNMF$fit$beta,Test$s)
    
    metric[i] = ctest[i]/rtest[i]
  }
  
  metrics = list(ctrain=ctrain,ctest=ctest,rtrain=rtrain,rmask=rmask,metric=metric)
  return(list(fits=fits,Htest=Htest,metrics=metrics,Tests=Tests,folds=folds))
}