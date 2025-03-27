#' @export
set_param_grid = function(k, alpha, lambda, eta, ninit, type, prefix,
                          ngene, maxit, tol, imaxit,
                          nfold=NULL, nmask=NULL, perc_mask=NULL){
  
  if(type=="full"){
    if(!dir.exists(paste0("results/",prefix,"/full/raw/"))){
      dir.create(paste0("results/",prefix,"/full/raw/"),recursive = TRUE)
    }
    params = expand.grid(k=k,alpha=alpha,lambda=lambda,eta=eta,lambdaW=lambdaW,lambdaH=lambdaH)
    params$file=paste0('results/',prefix,'/full/raw/',prefix,'_k=',params$k,
                       '_alpha',params$alpha,'_lambda',params$lambda,'_eta',params$eta,
                       '_lambdaW',params$lambdaW, '_lambdaH',params$lambdaH,
                       '_full','_ninit',ninit,
                       '_imaxit',imaxit,'_tol',tol,'_maxit',maxit,'.RData')
  }else if(type=="cv"){
    if(!dir.exists(paste0("results/",prefix,"/cv/raw/"))){
      dir.create(paste0("results/",prefix,"/cv/raw/"),recursive = TRUE)
    }
    params = expand.grid(k=k,alpha=alpha,lambda=lambda,eta=eta,fold=1:nfold)
    params$file=paste0('results/',prefix,'/cv/raw/',prefix,'_k=',params$k,
                       '_alpha',params$alpha,'_lambda',params$lambda,'_eta',params$eta,
                       '_fold',params$fold,'of',nfold,'_ninit',ninit,
                       '_imaxit',imaxit,'_tol',tol,'_maxit',maxit,'.RData')
  }else if(type=="mask"){
    if(!dir.exists(paste0("results/",prefix,"/mask/raw/"))){
      dir.create(paste0("results/",prefix,"/mask/raw/"),recursive = TRUE)
    }
    params = expand.grid(k=k,alpha=alpha,lambda=lambda,eta=eta,mask=1:nmask)
    params$file=paste0('results/',prefix,'/mask/raw/',prefix,'_k=',params$k,
                       '_alpha',params$alpha,'_lambda',params$lambda,'_eta',params$eta,
                       '_percmask',perc_mask,'_mask',params$mask,'_ninit',ninit,
                       '_imaxit',imaxit,'_tol',tol,'_maxit',maxit,'.RData')
  }
  
  # if(!replace){
  #   exists = numeric()
  #   i=1
  #   for(j in 1:nrow(params)){
  #     if(file.exists(params$file[j])){
  #       exists[i] = j
  #       i = i+1
  #     }
  #   }
  #   params = params[setdiff(1:nrow(params),exists),]
  # }
  
  return(params)
}