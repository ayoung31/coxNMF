set_param_grid = function(k, alpha, lambda, eta, ninit, replace, type, prefix,
                          ngene, maxit, tol, imaxit,
                          nfold=NULL, nmask=NULL, perc_mask=NULL){
  
  if(type=="full"){
    if(!dir.exists(paste0("results/",prefix,"/full"))){
      dir.create(paste0("results/",prefix,"/full"),recursive = TRUE)
    }
    params = expand.grid(k=k,alpha=alpha,lambda=lambda,eta=eta)
    params$file=paste0('results/',prefix,'/full/',prefix,'_k=',params$k,
                       '_alpha',params$alpha,'_lambda',params$lambda,'_eta',params$eta,
                       '_full','_ninit',ninit,
                       '_imaxit',imaxit,'_tol',tol,'_maxit',maxit,'.RData')
  }else if(type=="cv"){
    if(!dir.exists(paste0("results/",prefix,"/cv"))){
      dir.create(paste0("results/",prefix,"/cv"),recursive = TRUE)
    }
    params = expand.grid(k=k,alpha=alpha,lambda=lambda,eta=eta,fold=1:nfold)
    params$file=paste0('results/',prefix,'/cv/',prefix,'_k=',params$k,
                       '_alpha',params$alpha,'_lambda',params$lambda,'_eta',params$eta,
                       '_fold',params$fold,'of',nfold,'_ninit',ninit,
                       '_imaxit',imaxit,'_tol',tol,'_maxit',maxit,'.RData')
  }else if(type=="mask"){
    if(!dir.exists(paste0("results/",prefix,"/mask"))){
      dir.create(paste0("results/",prefix,"/mask"),recursive = TRUE)
    }
    params = expand.grid(k=k,alpha=alpha,lambda=lambda,eta=eta,mask=1:nmask)
    params$file=paste0('results/',prefix,'/mask/',prefix,'_k=',params$k,
                       '_alpha',params$alpha,'_lambda',params$lambda,'_eta',params$eta,
                       '_percmask',perc_mask,'_mask',params$mask,'_ninit',ninit,
                       '_imaxit',imaxit,'_tol',tol,'_maxit',maxit,'.RData')
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
  
  return(params)
}