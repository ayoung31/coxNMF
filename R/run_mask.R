#' @export
run_mask = function(X, y, delta, k, alpha, lambda=0, eta=0, perc_mask, nmask, 
                    ninit=100, imaxit=30, maxit=3000, tol=1e-5,
                    parallel=TRUE,ncore=NULL,replace=TRUE,
                    save=TRUE, verbose=TRUE, prefix){
  
  X=as.matrix(X)
  
  if(parallel & is.null(ncore)){
    ncore = detectCores() - 1
  }
  
  params = set_param_grid(k=k, alpha=alpha, lambda=lambda, eta=eta, ninit=ninit, 
                          replace=replace, type="mask", perc_mask=perc_mask, nmask=nmask, 
                          prefix=prefix, ngene = ngene, maxit=maxit, tol=tol, imaxit=imaxit)
  
  n=ncol(X)
  p=nrow(X)
  
  if(parallel){
    cl = parallel::makeCluster(ncore,outfile="")
    doParallel::registerDoParallel(cl)
    parallel::clusterCall(cl, function(x) .libPaths(x), .libPaths())
  }

  metrics = foreach(pa=1:nrow(params), .inorder = FALSE, .errorhandling = 'pass', 
                    .combine = 'rbind', .packages=c('coxNMF')) %dopar% {
    
    
    a = params$alpha[pa]
    l = params$lambda[pa]
    e = params$eta[pa]
    k = params$k[pa]
    m = params$mask[pa]
    
    set.seed(m)
    M = get_mask(perc_mask,n,p)
    
    fit_cox = run_coxNMF(X=X, y=y, delta=delta, k=k, M=M,
                         alpha=a, lambda=l, eta=e, 
                         tol=tol, maxit=maxit, verbose=verbose,
                         ninit=ninit, imaxit=imaxit)
    
    if(save){
      save(M,fit_cox,file=params$file[pa])
    }
    
    #compute masked recon error
    m_err = sum((1-M)*(X-fit_cox$W%*%fit_cox$H)^2)
    
    data.frame(k=k,alpha=a,lambda=l,eta=e,mask=m,mask_err=m_err)
    
  }#end foreach
  
  if(parallel){
    stopCluster(cl)
  }
  
  return(metrics)
  
}