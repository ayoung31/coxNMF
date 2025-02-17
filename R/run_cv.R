run_cv = function(X, y, delta, k, nfold, alpha, lambda = 0, eta = 0, ngene=5000,
                  ninit = 100, imaxit=30, maxit = 2000, tol = 1e-5, 
                  parallel = TRUE, ncore = NULL, 
                  replace = TRUE, save = TRUE, verbose=TRUE){
  
  X = as.matrix(X)
  
  if(parallel & is.null(ncore)){
    ncore = detectCores() - 1
  }
  
  params = set_param_grid(k=k, alpha=alpha, lambda=lambda, eta=eta, ninit=ninit, replace=replace, type="cv", nfold=nfold)

  
  fold_info = get_folds(X,y,nfold,ngene)
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
            .combine = 'rbind') %dopar% 
    {#begin foreach
    
    a = params$alpha[pa]
    l = params$lambda[pa]
    e = params$eta[pa]
    f = params$fold[pa]
    
    X = Xtrain[[f]]
    y_curr = y[folds!=f]
    delta_curr = delta[folds!=f]
    
    fit_cox = run_coxNMF(X=X, y=y_curr, delta=delta_curr, k=k, 
                         alpha=a, lambda=l, eta=e,
                         tol=tol, maxit=maxit, verbose=verbose,
                         ninit=ninit, imaxit=imaxit)
    if(save){
      save(fit_cox,file=params$file[pa])
    }
    
    #compute test set metrics: loss, sloss, recon error, cindex, bic
    ytest=y[folds==f]
    dtest=delta[folds==f]
    W = fit_cox$W
    H = fit_cox$beta
    beta = fit_cox$beta
    M = matrix(1,nrow=nrow(Xtest[[f]]),ncol=ncol(Xtest[[f]]))
    c = cvwrapr::getCindex(t(Xtest[[f]]) %*% W %*% beta, Surv(ytest, dtest))
    loss = calc_loss_cpp(Xtest[[f]], M, ytest, dtest, W, H, beta, a, l, e)
    ol = loss$loss
    sl = loss$surv_loss
    nl = loss$nmf_loss
    pen = loss$penalty
    bic = -2*sl + k*log(ncol(Xtest[[f]]))
    
    data.frame(alpha=a,lambda=l,eta=e,fold=f,loss=ol,sloss=sl,nloss=nl,pen=pen,bic=bic,c=c)
    
    
  }#end foreach
  
  if(parallel){
    stopCluster(cl)
  }

  return(metrics)

}
