#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat update_H_cpp(const arma::mat& X, const arma::mat& M, const arma::mat& W,
                       const arma::colvec& beta, const arma::mat& H, 
                       const arma::colvec& y, const arma::colvec& delta, 
                       double alpha, bool WtX) {
  arma::mat Hnew;
  arma::mat Wt = W.t();
  if(WtX){
    Hnew = H % (Wt * (M % X)) / (Wt * (M % (W * H)));
  }else{
    int N = H.n_cols;
    
    // linear predictor
    arma::vec lp = exp(H.t() * beta);
    
    // Indicator matrix
    arma::mat y_matrix = arma::repmat(y, 1, N);
    arma::mat I = arma::conv_to<arma::mat>::from(y_matrix >= y_matrix.t());
    
    // intermediate matrix
    arma::mat temp = I.t() % arma::repmat(lp.t(),N,1) / arma::repmat(I.t() * lp, 1, N);
    
    // derivative of log likelihood
    arma::rowvec delta_t = delta.t();
    arma::mat l = arma::kron(delta.t() - (delta.t() * temp),beta);
    
    // H update
    Hnew = (H / (Wt * (M % (W * H)))) % 
      ((Wt * (M % X)) + (alpha * arma::accu(M) / N) * 
      arma::clamp(l, 0, arma::datum::inf));
  }
  
  return Hnew;
}


// [[Rcpp::export]]
arma::mat update_W_cpp(const arma::mat& X, const arma::mat& M, const arma::mat& H,
                       const arma::mat& W, const arma::colvec& beta, 
                       const arma::colvec& y, const arma::colvec& delta, 
                       double alpha, bool WtX, int norm_type) {
  arma::mat Wnew;
  if(!WtX){
    Wnew = W % ((M % X) * H.t()) / ((M % (W * H)) * H.t());
  }else{
    //Need to update
  }
  
  
  if(norm_type == 1){
    arma::colvec row_sums = sum(Wnew, 1);
    Wnew.each_col() /= row_sums;
  }else if(norm_type == 2){
    arma::rowvec col_sums = sum(Wnew, 0);
    Wnew.each_row() /= col_sums;
  }
  
  return Wnew;
}

// [[Rcpp::export]]
List calc_loss_cpp(const arma::mat& X, const arma::mat& M, const arma::mat& W, const arma::mat& H,
               const arma::vec& beta, double alpha, const arma::vec& y, 
               const arma::vec& delta, double lambda, double eta, bool WtX) {
  int N = H.n_cols;
  
  double nmf_loss = arma::accu(arma::square(M % (X - W * H))) / arma::accu(M);
  
  arma::colvec a1;
  if (WtX) {
    a1 = trans(W.t() * (M % X)) * beta;
  } else {
    a1 = H.t() * beta;
  }

  arma::mat y_matrix = arma::repmat(y, 1, N);
  arma::mat ind = arma::conv_to<arma::mat>::from(y_matrix >= y_matrix.t());

  double surv_loss = 2 * arma::accu(delta % (a1 - arma::log(ind.t() * arma::exp(a1)))) / N;
  double penalty = lambda * ((1 - eta) * arma::accu(arma::square(beta)) / 2 + eta * arma::accu(arma::abs(beta)));
  double loss = nmf_loss - alpha * (surv_loss - penalty);
  
  return List::create(
    Named("loss") = loss,
    Named("nmf_loss") = nmf_loss,
    Named("surv_loss") = surv_loss
  );
}

// 
// // [[Rcpp::export]]
// List optimize_loss_cpp(const arma::mat& X, const arma::mat& M,
//                             const arma::mat& H0, const arma::mat& W0,
//                             const arma::colvec& beta0, const arma::colvec& y,
//                             const arma::colvec& delta, double alpha,
//                             double lambda, double eta, double tol,
//                             int maxit, bool verbose, bool WtX, int norm_type){
//   arma::mat H = H0;
//   arma::mat W = W0;
//   arma::colvec beta = beta0;
//   double loss = 0;
//   double eps = 1;
//   int it = 0;
// 
//   while(eps > tol && it <= maxit){
//     loss_prev = loss;
//     eps_prev = eps;
//     
//     W = update_W_cpp(X,M,H,W,beta,y,delta,alpha,WtX,norm_type);
//     H = update_H_cpp(X,M,W,beta,H,y,delta,alpha,WtX);
//     beta = 
//   }
// 
//   List L = List::create(W,H,beta);
//   return L;
// }