#' @export
run_full_warm_start = function(X, y, delta, k, alpha, lambda = 0, eta = 0,
                    ninit = 100, imaxit = 30, maxit = 3000, tol = 1e-5, 
                    parallel = TRUE, ncore = NULL, replace = FALSE, 
                    save = TRUE, verbose=TRUE, prefix){
  
  X=as.matrix(X)
  
  if(ncore==1){
    parallel=FALSE
  }
  
  if(parallel & is.null(ncore)){
    ncore = detectCores() - 1
  }
  
  params = set_param_grid(k=k, alpha=alpha, lambda=lambda, eta=eta, ninit=ninit, 
                          type="full", prefix=prefix,
                          ngene = ngene, maxit=maxit, tol=tol, imaxit=imaxit)
  
  params2=params %>% filter(alpha==0)
  
  if(parallel){
    cl = parallel::makeCluster(ncore,outfile="")
    doParallel::registerDoParallel(cl)
    parallel::clusterCall(cl, function(x) .libPaths(x), .libPaths())
  }
  
  metrics = 
    foreach(pa=1:nrow(params2), .inorder = FALSE, .errorhandling = 'remove', 
            .combine = 'rbind', .packages = c("coxNMF","survival","cvwrapr")) %dopar% {
    
    #a = params$alpha[pa]
    l = params$lambda[pa]
    e = params$eta[pa]
    k = params$k[pa]
    
    dat = list()
    i=1
    for(a in alpha){
      file = params$file[params$alpha==a & params$lambda==l & params$eta==e & params$k==k]
      if(replace | !file.exists(file)){
        print(sprintf('alpha=%f, lambda=%f, eta=%f, k=%d\n',a,l,e,k))
        print("running...")
        if(a==0){
          fit_cox = run_coxNMF(X=X, y=y, delta=delta, k=k, 
                               alpha=a, lambda=l, eta=e, 
                               tol=tol, maxit=maxit, verbose=verbose,
                               ninit=ninit, imaxit=imaxit)
        }else{
          fit_cox = run_coxNMF(X=X, y=y, delta=delta, k=k, 
                               alpha=a, lambda=l, eta=e, 
                               tol=tol, maxit=maxit, verbose=verbose,
                               ninit=ninit, imaxit=imaxit,
                               W0=fit_cox$W, H0=fit_cox$H, beta0=fit_cox$beta)
        }
        
        if(save){
          save(fit_cox,file=file)
        }
      }else{
        #print("loading...")
        load(file)
      }
      
      if(fit_cox$`NaN flag`){
        break
      }
      
      
      #primary metrics to output
      M=matrix(1,nrow=nrow(X),ncol=ncol(X))
      W = fit_cox$W
      H = fit_cox$H
      beta = fit_cox$beta
      c = cvwrapr::getCindex(t(X) %*% W %*% beta, Surv(y, delta))
      loss = fit_cox$loss
      ol = loss$loss
      sl = loss$surv_loss
      nl = loss$nmf_loss
      pen = loss$penalty
      bic = -2*sl + k*log(ncol(X))
      converged=fit_cox$iter < maxit
      
      dat[[i]] = data.frame(k=k,alpha=a,lambda=l,eta=e,c=c,loss=ol,sloss=sl,
                 nloss=nl,pen=pen,bic=bic,converged=converged,niter=fit_cox$iter,
                 flag_nan = fit_cox$`NaN flag`)
      i=i+1
      
      
      
    }#end alpha loop

    do.call('rbind',dat)
  }#end foreach
  
  if(parallel){
    stopCluster(cl)
  }
  
  return(metrics)
  
}
