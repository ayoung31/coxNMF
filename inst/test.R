library(coxNMF)
library(dplyr)
library(NMF)
library(pheatmap)

k=5
#read pre-filtered data
X= readRDS('data/filtered_subfraction_only.rds')
X=as.matrix(X)
#X = sqrt(X)
y=rep(0,ncol(X))
delta=rep(0,ncol(X))
Train=list()
Train$X=X


n=ncol(X)
p=nrow(X)


#initialize parameters
set.seed(86)
H0 = matrix(runif(n*k,0,max(X)),nrow=k)
W0 = matrix(runif(p*k,0,max(X)),nrow=p)
beta0 = rep(0,k)#runif(k,-.000001,.000001)
M = get_mask(Train,.3)
#H0 = sweep(H0,2,colSums(H0),'/')
# init = nmfModel(k,X,W=W0,H=H0)
# fit_std = nmf(X,k,"lee",seed=init,.options="v10",)

fit_cox = run_coxNMF(X=X,y=y,delta=delta,k=k,alpha=0,lambda=0,eta=0,H0=H0,
                     W0=W0,beta0=beta0,M=M,tol=1e-7,maxit=2000,verbose=TRUE,WtX=TRUE)

#ra = recommend_alpha(X,M,y,delta,k,10,WtX=TRUE,norm.type = 2)

cvwrapr::getCindex(t(X)%*%fit_std@fit@W%*%fit_cox$beta,survival::Surv(y,delta))
cvwrapr::getCindex(t(X)%*%fit_cox$W%*%fit_cox$beta,survival::Surv(y,delta))

pheatmap(cor(t(rbind(fit_std@fit@H,fit_cox$H))),cluster_rows = FALSE,cluster_cols = FALSE)
pheatmap(cor(cbind(fit_std@fit@W,fit_cox$W)))


          
load('data/cmbSubtypes_formatted.RData')
source('data/helper_functions.R')
### genes
colors <- c('orange','blue','pink',
            'green','yellow','purple','red')
text <- c('black','white','black','black','black','white','white')


i=3
names(top_genes)[i]
print(paste(colors,names(top_genes[[i]]),sep='='))     

apply(t(fit_cox$W)%*%X,1,sd)*fit_cox$beta

W <- fit_cox$W

rownames(W) = rownames(X)
tops <- get_top_genes(W,ngene=25)
create_table(tops,top_genes[[i]])


Hstd = apply(fit_std@fit@H,1,sd)
Hm = apply(fit_std@fit@H,1,mean)

H = sweep(fit_std@fit@H,1,Hm,"-")
H = sweep(H,1,Hstd,"/")
stddat = data.frame(t(H))
colnames(stddat) = paste0('H',1:k)
stddat$y = y
stddat$delta = delta
fit=survival::coxph(survival::Surv(y,delta)~H1+H2+H3+H4+H5+H6+H7+H8+H9+H10,stddat)


boxplot(fit_cox$H[10,] ~ tcga$sampInfo$PurIST)
boxplot(fit_std@fit@H[10,] ~ tcga$sampInfo$PurIST)
