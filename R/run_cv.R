
run_cv = function(X,y,delta,k,nfold,alpha,lambda,eta,ninit,parallel=TRUE,ncore=NULL,replace=TRUE){
  
  if(parallel & is.null(ncore)){
    ncore = detectCores() - 1
  }
  
  if(!replace){
    exists = numeric()
    i=1
    for(j in 1:nrow(params)){
      if(file.exists(params$file[j])){
        exists[i] = j
        i = i+1
      }
    }
    params = params[setdiff(1:nrow(params),exists),]
  }

  params = expand.grid(alpha=alpha,lambda=lambda,eta=eta,gamma=gamma,fold=1:nfold)
  
  params$file=paste0('results/res_k=',k,'_alpha',params$alpha,'_lambda',params$lambda,'_eta',params$eta,'_gamma',params$gamma,'_fold',params$fold,'of',nfold,'_ninit',ninit,'.RData')
  
  if(parallel){
    cl = parallel::makeCluster(ncore,outfile="")
    doParallel::registerDoParallel(cl)
    parallel::clusterCall(cl, function(x) .libPaths(x), .libPaths())
  }
  
  
  foreach(pa=1:nrow(params), .inorder = FALSE, .errorhandling = 'pass', .combine = 'rbind') %dopar% {
    
    a = params$alpha[pa]
    l = params$lambda[pa]
    e = params$eta[pa]
    g = params$gamma[pa]
    f = params$fold[pa]
    
    X = Xtrain[[f]]
    y = sampInfo$Follow.up.days[folds!=f]
    delta = -1*(as.numeric(sampInfo$Censored.1yes.0no[folds != f])-2)
    
    n=ncol(X)
    p=nrow(X)
    
    M = matrix(1,ncol=n,nrow=p)
    
    min_loss = Inf
    for(init in 1:ninit){
      #initialize parameters
      set.seed(init)
      H0 = matrix(runif(n*k,0,max(X)),nrow=k) # need to change matrix size here
      W0 = matrix(runif(p*k,0,max(X)),nrow=p)
      beta0 = rep(0,k)
      
      print(sprintf("pa: %d init: %d",pa,init))
      fit_curr = coxNMF::run_coxNMF(X,y,delta,k,a, l,e,H0,W0,beta0,tol=1e-6,
                                    maxit=15,verbose=TRUE,WtX=TRUE,gamma=g)
      if(fit_curr$loss$loss < min_loss){
        min_loss = fit_curr$loss$loss
        best = fit_curr
      }
    }
    
    fit_cox = coxNMF::run_coxNMF(X,y,delta,k,a, l,e,best$H,best$W,best$beta,tol=1e-6,
                                 maxit=3000,verbose=TRUE,WtX=TRUE)
    save(fit_cox,file=params$file[pa])
    
  }
  
  if(parallel){
    stopCluster(cl)
  }

}
