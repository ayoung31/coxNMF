#' @export
run_cv = function(X, y, delta, k, nfold, alpha, lambda = 0, eta = 0, fold_info,
                  ninit = 100, imaxit=30, maxit = 3000, tol = 1e-5, 
                  parallel = TRUE, ncore = NULL, replace = FALSE, 
                  save = TRUE, verbose=TRUE, prefix){
  
  X = as.matrix(X)
  
  if(parallel & is.null(ncore)){
    ncore = detectCores() - 1
  }
  
  params = set_param_grid(k=k, alpha=alpha, lambda=lambda, eta=eta, ninit=ninit, 
                          type="cv", nfold=nfold, prefix = prefix,
                          ngene = ngene, maxit=maxit, tol=tol, imaxit=imaxit)
  
  Xtrain = fold_info$Xtrain
  Xtest = fold_info$Xtest
  folds = fold_info$folds
  
  if(parallel){
    cl = parallel::makeCluster(ncore,outfile="")
    doParallel::registerDoParallel(cl)
    parallel::clusterCall(cl, function(x) .libPaths(x), .libPaths())
  }
  
  metrics = 
    foreach(pa=1:nrow(params), 
            .inorder = FALSE, 
            .errorhandling = 'pass', 
            .combine = 'rbind',
            .packages = c("coxNMF","survival","cvwrapr")) %dopar% 
    {#begin foreach
    
    a = params$alpha[pa]
    l = params$lambda[pa]
    e = params$eta[pa]
    f = params$fold[pa]
    
    X = Xtrain[[f]]
    y_curr = y[folds!=f]
    delta_curr = delta[folds!=f]
    
    if(replace | !file.exists(params$file[pa])){
      fit_cox = run_coxNMF(X=X, y=y_curr, delta=delta_curr, k=k, 
                           alpha=a, lambda=l, eta=e,
                           tol=tol, maxit=maxit, verbose=verbose,
                           ninit=ninit, imaxit=imaxit)
    }else{
      load(params$file[pa])
    }
    
    if(save){
      save(fit_cox,file=params$file[pa])
    }
    
    #compute test set metrics: loss, sloss, recon error, cindex, bic
    ytest=y[folds==f]
    dtest=delta[folds==f]
    W = fit_cox$W
    H = fit_cox$H
    beta = fit_cox$beta
    M = matrix(1,nrow=nrow(Xtest[[f]]),ncol=ncol(Xtest[[f]]))
    c = cvwrapr::getCindex(t(Xtest[[f]]) %*% W %*% beta, Surv(ytest, dtest))
    sl = calc_surv_loss(Xtest[[f]], M, ytest, dtest, W, beta)
    bic = -2*sl + k*log(ncol(Xtest[[f]]))
    
    converged=fit_cox$iter < maxit
    
    data.frame(k=k,alpha=a,lambda=l,eta=e,fold=f,sloss=sl,bic=bic,c=c, converged=converged)
    
    
  }#end foreach
  
  if(parallel){
    stopCluster(cl)
  }

  return(metrics)

}
