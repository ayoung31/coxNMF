#' @export
get_folds = function(n,nfold){
  rand = sample(n)
  folds = list()
  for(i in 1:nfold){
    folds[[i]] = rand[rand %% nfold + 1 == i]
  }
  return(folds)
}
