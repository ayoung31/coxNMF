#' @export
cv.coxNMF = function(X, y, delta, k, alpha, lambda, eta, WtX = FALSE,
                      verbose = FALSE, norm_type = 2, tol = 1e-6, maxit = 10000,
                      penalty = 'lasso', nfold = 5, perc_miss = .3, seed = 123,...){
  
  set.seed(seed)
  folds = get_folds(ncol(X), nfold)
  
  fits = list()
  Hval = list()
  cval = numeric()
  ctrain = numeric()
  rtrain = numeric()
  rmask = numeric()
  Trains = list()
  Vals = list()
  metric = numeric()
  for(i in 1:nfold){
    Train = list()
    Val = list()
    Train$X = X[,-folds[[i]]]
    Train$y = y[-folds[[i]]]
    Train$delta = delta[-folds[[i]]]
    Val$X = X[,folds[[i]]]
    Val$y = y[folds[[i]]]
    Val$delta = delta[folds[[i]]]
    
    # within Train$X sample ~30% of cells to be missing
    Train$M = get_mask(Train, perc_miss)
    
    # save train and val datasets
    Trains[[i]] = Train
    Vals[[i]] = Val
    
    # Run the model
    coxNMF = run_coxNMF(Train$X, Train$y, Train$delta, k, alpha, lambda, eta, 
                        Train$M, WtX, verbose, norm_type, tol, maxit, penalty,...)
    
    # save the fit
    fits[[i]] = coxNMF
    
    ## Compute metrics
    
    # reconstruction error
    rtrain[i] = coxNMF$loss$nmf_loss
    if(sum(1-Train$M)!=0){
      rmask[i] = norm((1-Train$M)*(Train$X-coxNMF$W%*%coxNMF$H),'F')^2 / sum(1-Train$M)
    }else{
      rmask[i] = NA # if we didn't do any masking there is no rmask
    }
    
    # cindex
    if(WtX){
      ctrain[i] = cvwrapr::getCindex(t(t(coxNMF$W)%*%Train$X)%*%coxNMF$beta,Surv(Train$y,Train$delta))
      Hval[[i]] = t(coxNMF$W)%*%Val$X # Hval is irrelevant for WtX version
    }else{
      ctrain[i] = cvwrapr::getCindex(t(coxNMF$H)%*%coxNMF$beta,Surv(Train$y,Train$delta))
      Hval[[i]] = NMF::.fcnnls(coxNMF$W,Val$X)$coef
    }
    cval[i] = cvwrapr::getCindex(t(Hval[[i]])%*%coxNMF$beta,Surv(Val$y,Val$delta))
    
    # our metric
    metric[i] = cval[i]/rmask[i]
    
    print(sprintf('Fold %d complete', i))
    
  }
  
  metrics = list(ctrain=ctrain,cval=cval,rtrain=rtrain,rmask=rmask,metric=metric)
  return(list(fits=fits,Hval=Hval,metrics=metrics,Vals=Vals))
}