

run_mask = function(X, k, alpha, lambda=0, eta=0, 
                    perc_mask=.3, nmask=5, 
                    ninit=100, imaxit=30,
                    parallel=TRUE,ncore=NULL,replace=TRUE,
                    save=TRUE, verbose=TRUE){
  
  X=as.matrix()
  
  if(parallel & is.null(ncore)){
    ncore = detectCores() - 1
  }
  
  params = set_param_grid(k=k, alpha=alpha, lambda=lambda, eta=eta, ninit=ninit, replace=replace, type="mask", perc_mask=perc_mask, nmask=nmask)
  
  y = rep(0,ncol(X))
  delta = rep(0,ncol(X))
  n=ncol(X)
  p=nrow(X)
  
  ncore <- 50#### CHANGE to 60
  
  if(parallel){
    cl = parallel::makeCluster(ncore,outfile="")
    doParallel::registerDoParallel(cl)
    parallel::clusterCall(cl, function(x) .libPaths(x), .libPaths())
  }

  metrics = foreach(pa=1:nrow(params), .inorder = FALSE, .errorhandling = 'pass', .combine = 'rbind') %dopar% {
    
    
    a = params$alpha[pa]
    l = params$lambda[pa]
    e = params$eta[pa]
    k = params$k[pa]
    m = params$mask[pa]
    
    set.seed(m)
    M = coxNMF::get_mask(perc_mask,n,p)
    
    fit_cox = coxNMF::run_coxNMF(X=X, y=y, delta=delta, k=k, M=M,
                                 alpha=a, lambda=l, eta=e, 
                                 tol=tol, maxit=maxit, verbose=verbose,
                                 ninit=ninit, imaxit=imaxit)
    
    if(save){
      save(M,fit_cox,file=params$file[pa])
    }
    
    #compute masked recon error
    sum((1-M)*(X-fit_cox$W%*%fit_cox$H)^2)
    
  }#end foreach
  
  if(parallel){
    stopCluster(cl)
  }
  
  return(metrics)
  
}