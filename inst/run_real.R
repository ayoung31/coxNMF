library(coxNMF)
library(foreach)
library(dplyr)
library(parallel)

k=as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
ninit = 100

#read pre-filtered data
tcga = readRDS('TCGA_PAAD_gencode_filtered.rds')

X=as.matrix(tcga$ex)
y=tcga$sampInfo$Follow.up.days
delta=-1*(as.numeric(tcga$sampInfo$Censored.1yes.0no)-2)

M = matrix(1,ncol=ncol(X),nrow=nrow(X))
ra = recommend_alpha(X,M,y,delta,k,10,WtX=FALSE,norm.type = 2)

# alpha = seq(0,ra$alpha5050,length.out=20)#c(0,ra$alpha_grid[1:6])#c(0,ra$alpha5050*(10^seq(-5,1)))
# alpha=alpha[1:10]
alpha=seq(4e5,8e5,by=2e4)#seq(0,3000,by=100)#seq(185201.0143700,185201.0143750,by=.0000005)#seq(185201.01435,185201.01440,by=.000005)#seq(185201.01400,185201.01450,by=.00005)#seq(185201.01,185201.02,by=.0005)#seq(185201.0,185201.5,by=.01)#seq(185200,185210,by=.5)#seq(185000,185500,by=10)#seq(1.8e5,2e5,by=500)#seq(0,4e5,by=2e4)

lambda=.1#c(.1,.5)#c(.1,.5,1)
eta=.1#c(.1,.5,.9)

n=ncol(X)
p=nrow(X)


ncore <-100 #### CHANGE to 60

cl = parallel::makeCluster(ncore)
doParallel::registerDoParallel(cl)
parallel::clusterCall(cl, function(x) .libPaths(x), .libPaths())

metrics = foreach(a=alpha, .inorder = FALSE, .errorhandling = 'pass', .combine='rbind') %:%
  foreach(init=1:ninit, .inorder = FALSE, .errorhandling = 'pass', .combine='rbind') %:%
  foreach(l=lambda, .inorder=FALSE, .errorhandling = 'pass', .combine='rbind') %:%
    foreach(e=eta, .inorder=FALSE, .errorhandling = 'pass', .combine='rbind') %dopar% {
      
      #initialize parameters
      set.seed(init)
      H0 = matrix(runif(n*k,0,max(X)),nrow=k)
      W0 = matrix(runif(p*k,0,max(X)),nrow=p)
      beta0 = runif(k,-1,1)
      
      print(a)
      fit_cox = coxNMF::run_coxNMF(X,y,delta,k,a, l,e,H0,W0,beta0,tol=1e-6,maxit=100,verbose=FALSE)
      
      dl = data.frame(k=k,
                   alpha=a,
                   lambda=l,
                   eta=e,
                   init=init,
                   loss=fit_cox$loss$loss)
      
      dl
          
        
}

best = metrics %>% group_by(alpha,lambda,eta) %>% summarise(lo=min(loss),init=init[which.min(loss)])
1:nrow(best)
save(metrics,file=paste0('results/metrics_k=',k,'.RData'))
save(best,file=paste0('results/best_k=',k,'.RData'))



foreach(a=alpha, .inorder = FALSE, .errorhandling = 'pass', .combine='rbind', .packages = "dplyr") %:%
  foreach(l=lambda, .inorder=FALSE, .errorhandling = 'pass', .combine='rbind', .packages = "dplyr") %:%
  foreach(e=eta, .inorder=FALSE, .errorhandling = 'pass', .combine='rbind', .packages = "dplyr") %dopar% {
    
    curr = best %>% filter(alpha==a & lambda==l & eta==e)
    set.seed(curr$init)
    
    H0 = matrix(runif(n*k,0,max(X)),nrow=k)
    W0 = matrix(runif(p*k,0,max(X)),nrow=p)
    beta0 = runif(k,-1,1)
    
    print(a)
    fit_cox = coxNMF::run_coxNMF(X,y,delta,k,a, l,e,H0,W0,beta0,tol=1e-6,maxit=2000,verbose=FALSE)
    
    save(fit_cox,file=paste0('results/res_k=',k,'_alpha',a,'_lambda',l,'_eta',e,'.RData'))
}
    