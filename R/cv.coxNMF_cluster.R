
# read job array number
array_num <- Sys.getenv("SLURM_ARRAY_TASK_ID")

# read sim # as parameter passed from sbatch
args <- commandArgs(trailingOnly=TRUE)
sim <- as.numeric(args[1])

# load params 1
load('params/params1.RData')

# load params 2
load('params/params2.RData')

# load data
load(paste0(load('data/sim_',sim,'.RData')))

# get k and alpha from params 1 based on array job number
k = params1$k[array_num]
alpha = params1$alpha[array_num]

# set seed based on sim number
set.seed(sim)

# create output folders if they don't exist
if(!dir.exists('folds')){
  dir.create('folds')
}
if(!dir.exists('masks')){
  dir.create('masks')
}
if(!dir.exists('results')){
  dir.create('results')
}


folds = get_folds(ncol(X), nfold) # should probably save folds at least
save(folds,file=paste0("folds/sim",sim,"_k=",k,"_alpha=",alpha,".RData"))

fits = list()
Hval = list()
cval = numeric()
ctrain = numeric()
rtrain = numeric()
rmask = numeric()
ltrain = numeric()
lval = numeric()
strain = numeric()
sval = numeric()
Trains = list()
Vals = list()
metric = numeric()

metrics = data.frame(matrix(nrow=0,ncol=11))
colnames(metrics)=c('fold','lambda','eta','ctrain','cval','rtrain','rmask',
                    'strain','sval','ltrain','lval')
j = 1
for(i in 1:nfold){
  Train = list()
  Val = list()
  Train$X = X[,-folds[[i]]]
  Train$y = y[-folds[[i]]]
  Train$delta = delta[-folds[[i]]]
  Val$X = X[,folds[[i]]]
  Val$y = y[folds[[i]]]
  Val$delta = delta[folds[[i]]]

  # within Train$X sample ~30% of cells to be missing
  Train$M = get_mask(Train, perc_miss)
  # save mask
  save(Train$M,file="masks/sim_",sim,"_k=",k,"_alpha=",alpha,".RData")

  # add lambda and eta loop here
  for(l in 1:nrow(params2)){
    lambda = params2$lambda[l]
    eta = params2$lambda[l]
    # Run the model
    coxNMF = run_coxNMF(Train$X, Train$y, Train$delta, k, alpha, lambda, eta,
                        Train$M, WtX, verbose, norm_type, tol, maxit, penalty)

    # save the fit
    fits[[j]] = coxNMF

    ## Compute metrics

    # reconstruction error
    rtrain[j] = coxNMF$loss$nmf_loss
    if(sum(1-Train$M)!=0){
      rmask[j] = norm((1-Train$M)*(Train$X-coxNMF$W%*%coxNMF$H),'F')^2 / sum(1-Train$M)
    }else{
      rmask[j] = NA # if we didn't do any masking there is no rmask
    }

    # cindex
    if(WtX){
      ctrain[j] = cvwrapr::getCindex(t(t(coxNMF$W)%*%Train$X)%*%coxNMF$beta,Surv(Train$y,Train$delta))
      Hval[[j]] = t(coxNMF$W)%*%Val$X # Hval is irrelevant for WtX version
    }else{
      ctrain[j] = cvwrapr::getCindex(t(coxNMF$H)%*%coxNMF$beta,Surv(Train$y,Train$delta))
      Hval[[j]] = NMF::.fcnnls(coxNMF$W,Val$X)$coef
    }
    cval[j] = cvwrapr::getCindex(t(Hval[[j]])%*%coxNMF$beta,Surv(Val$y,Val$delta))

    # loss
    strain[j] = coxNMF$loss$surv_loss
    sval[j] = calc_surv_loss(Val$X,coxNMF$W,Hval[[j]],coxNMF$beta,Val$y,Val$delta,FALSE)
    
    ltrain[j] = coxNMF$loss$loss
    lval[j] = rmask[j] - alpha * sval[j]

    # our metric
    metric[j] = cval[j]/rmask[j]

    # add all metrics to data frame here
    metrics[j,] = c(i,lambda,eta,ctrain,cval,rtrain,rmask,strain,sval,ltrain,lval)

    # print lambda, eta or progress bar
    print(sprintf('Lambda: %.2f - Eta: %.2f - Parameter combo %d/%d',
                  lambda,eta,l,nrow(params2)))
    j = j + 1

  }# end lambda eta loop here

  print(sprintf('Fold %d complete', i))
}

# write data frame to .RData file with name sim, k, alpha
save(metrics,file=paste0('results/sim_',sim,'_k=',k,'_alpha=',alpha,'.RData'))
