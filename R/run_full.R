run_full = function(X, y, delta, k, alpha, lambda = 0, eta = 0, ngene = 5000, 
                    ninit = 100, imaxit = 30, maxit = 2000, tol = 1e-5, 
                    parallel = TRUE, ncore = NULL, 
                    replace = TRUE, save = TRUE, verbose=TRUE){
  
  X=as.matrix(X)
  
  if(parallel & is.null(ncore)){
    ncore = detectCores() - 1
  }
  
  params = set_param_grid(k=k, alpha=alpha, lambda=lambda, eta=eta, ninit=ninit, replace=replace, type="full")

  X = gene_filter(X=X, ngene=ngene)
  
  if(parallel){
    cl = parallel::makeCluster(ncore,outfile="")
    doParallel::registerDoParallel(cl)
    parallel::clusterCall(cl, function(x) .libPaths(x), .libPaths())
  }
  
  foreach(pa=1:nrow(params), .inorder = FALSE, .errorhandling = 'pass', .combine = 'rbind') %dopar% {
    
    a = params$alpha[pa]
    l = params$lambda[pa]
    e = params$eta[pa]
    
    fit_cox = run_coxNMF(X=X, y=y, delta=delta, k=k, 
                         alpha=a, lambda=l, eta=e, 
                         tol=tol, maxit=maxit, verbose=verbose,
                         ninit=ninit, imaxit=imaxit)
    
    save(fit_cox,file=params$file[pa])
    
  }#end foreach
  
  if(parallel){
    stopCluster(cl)
  }
  
}