#' @export
get_mask <- function(Train,perc_miss){
  n_train <- ncol(Train$X)
  p_train <- nrow(Train$X)
  M <- matrix(1,nrow=p_train,ncol=n_train)
  n_miss <- floor(n_train*p_train*perc_miss)
  entries <- as.vector(outer(1:n_train, 1:p_train, paste, sep="_"))
  miss_entries <- sample(entries,n_miss,replace=FALSE)
  rcs <- strsplit(miss_entries,'_')
  cols <- as.numeric(lapply(rcs,'[',1))
  rows <- as.numeric(lapply(rcs,'[',2))
  M[cbind(rows,cols)] <- 0
  
  return(M)
}
