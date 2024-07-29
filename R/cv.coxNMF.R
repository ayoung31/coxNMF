#' @importFrom foreach %dopar%
#' @export
cv.coxNMF = function(X, y, delta, k, alpha, lambda, eta, WtX = FALSE,
                     verbose = FALSE, norm_type = 2, tol = 1e-6, maxit = 10000,
                     penalty = 'lasso', nfold = 5, perc_miss = .3, seed = 123,
                     ncore = NULL, ...){
  
  if(is.null(ncore)){
    ncore = detectCores() - 1
  }
  
  set.seed(seed)
  folds = get_folds(ncol(X), nfold)
  
  cl = parallel::makeCluster(ncore)
  doParallel::registerDoParallel(cl)
  parallel::clusterCall(cl, function(x) .libPaths(x), .libPaths())
  
  metrics = list()
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
    
    
    # start loop over k, alpha, eta here
    metrics[[i]] = foreach(K=k, .inorder = FALSE, .combine = 'rbind', .errorhandling = 'remove') %:%
      foreach(a=alpha, .inorder = FALSE, .combine = 'rbind', .errorhandling = 'remove') %:%
        foreach(e=eta, .inorder = FALSE, .combine = 'rbind', .errorhandling = 'remove') %dopar% {
          # initialization
          lam = lambda[order(lambda)]
          if(lam[1] != 0){
            lam = c(0,lam)
          }
          if(a==0){
            lam=0
          }
          
          
          rtrain = numeric()
          rmask = numeric()
          ctrain = numeric()
          cval = numeric()
          strain = numeric()
          sval = numeric()
          ltrain = numeric()
          lval = numeric()
          metric = numeric()
          j = 1
          # warm start lambda
          for(l in lam){
            if(l==0){
              coxNMF = run_coxNMF(Train$X, Train$y, Train$delta, K, a, l, e, 
                                  M=Train$M, WtX=WtX, verbose=verbose, 
                                  norm_type=norm_type, tol=tol, maxit=maxit, 
                                  penalty=penalty,ninit=10)
            }else{
              coxNMF = optimize_loss_cpp(Train$X, Train$M, coxNMF$H, coxNMF$W, 
                                         coxNMF$beta, Train$y, Train$delta, a, 
                                         l, e, tol, maxit, verbose, WtX, 
                                         norm_type, penalty, FALSE)
            }
            
            
            ## Compute metrics
            
            # reconstruction error
            rtrain[j] = coxNMF$loss$nmf_loss
            if(sum(1-Train$M)!=0){
              rmask[j] = norm((1-Train$M)*(Train$X-coxNMF$W%*%coxNMF$H),'F')^2 / sum(1-Train$M)
            }else{
              rmask[j] = NA # if we didn't do any masking there is no rmask
            }
            
            # cindex
            if(WtX){
              ctrain[j] = cvwrapr::getCindex(t(t(coxNMF$W)%*%Train$X)%*%coxNMF$beta,survival::Surv(Train$y,Train$delta))
              Hval = t(coxNMF$W)%*%Val$X # Hval is irrelevant for WtX version
            }else{
              ctrain[j] = cvwrapr::getCindex(t(coxNMF$H)%*%coxNMF$beta,survival::Surv(Train$y,Train$delta))
              Hval = NMF::.fcnnls(coxNMF$W,Val$X)$coef
            }
            cval[j] = cvwrapr::getCindex(t(Hval)%*%coxNMF$beta,survival::Surv(Val$y,Val$delta))
            
            # loss
            strain[j] = coxNMF$loss$surv_loss
            sval[j] = calc_surv_loss(Val$X,coxNMF$W,Hval,coxNMF$beta,Val$y,Val$delta,FALSE)
            
            ltrain[j] = coxNMF$loss$loss
            lval[j] = rmask[j] - alpha * sval[j]
            
            # our metric
            metric[j] = cval[j]/rmask[j]
            
            j = j+1
          }# end warm start lambda
          
          data.frame(rtrain=rtrain, rmask=rmask, ctrain=ctrain, cval=cval,
                     strain=strain, sval=sval, ltrain=ltrain, lval=lval, 
                     metric = metric, lambda = lam, eta = e, alpha = a, 
                     k = K, fold = i)
        }
    
    
      
    print(sprintf('Fold %d complete', i))
    
  }
  parallel::stopCluster(cl)
  
  metrics = do.call('rbind',metrics)
  
  met_mean = metrics %>% group_by(k,alpha,lambda,eta) %>%
    summarise(met = mean(metric))
  
  k = met_mean$k[which.max(met_mean$met)]
  alpha = met_mean$alpha[which.max(met_mean$met)]
  lambda = met_mean$lambda[which.max(met_mean$met)]
  eta = met_mean$eta[which.max(met_mean$met)]
  
  fit = run_coxNMF(X,y,delta,k,alpha,lambda,eta,ninit)
  
  return(list(metrics=metrics, final_fit=fit))
}