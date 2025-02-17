set_param_grid = function(k, alpha, lambda, eta, ninit, replace, type, nfold=NULL, nmask=NULL, perc_mask=NULL){
  
  if(type=="full"){
    params = expand.grid(k=k,alpha=alpha,lambda=lambda,eta=eta)
    params$file=paste0('results/res_k=',params$k,'_alpha',params$alpha,'_lambda',params$lambda,'_eta',params$eta,'_full','_ninit',ninit,'.RData')
  }else if(type=="cv"){
    params = expand.grid(k=k,alpha=alpha,lambda=lambda,eta=eta,fold=1:nfold)
    params$file=paste0('results/res_k=',params$k,'_alpha',params$alpha,'_lambda',params$lambda,'_eta',params$eta,'_fold',params$fold,'of',nfold,'_ninit',ninit,'.RData')
  }else if(type=="mask"){
    params = expand.grid(k=k,alpha=alpha,lambda=lambda,eta=eta,mask=1:nmask)
    params$file=paste0('results/res_k=',params$k,'_alpha',params$alpha,'_lambda',params$lambda,'_eta',params$eta,'_percmask',perc_mask,'_mask',params$mask,'_ninit',ninit,'.RData')
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