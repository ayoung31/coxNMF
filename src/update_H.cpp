#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
void update_H(const arma::mat& X, const arma::mat& M, const arma::mat& W,
                   const arma::colvec& beta, arma::mat& H, 
                   const arma::colvec& y, const arma::colvec& delta, 
                   double alpha, bool WtX) {
  arma::mat Wt = W.t();
  if(WtX){
    H = H % (Wt * (M % X)) / (Wt * (M % (W * H)));
  }else{
    int N = H.n_cols;
    
    // linear predictor
    arma::vec lp = exp(H.t() * beta);
    
    // Indicator matrix
    arma::mat y_matrix = arma::repmat(y, 1, N);
    arma::mat I = arma::conv_to<arma::mat>::from(y_matrix == y_matrix.t());
    
    // intermediate matrix
    arma::mat temp = I.t() % arma::repmat(I.t() * lp, 1, N) % arma::repmat(lp.t(),N,1);
    
    // derivative of log likelihood
    arma::rowvec delta_t = delta.t();
    arma::mat l = arma::kron(delta.t() - (delta.t() * temp),beta);
    
    // H update
    H = (H / (Wt * (M % (W * H)))) % 
      ((Wt * (M % X)) + (alpha * arma::accu(M) / N) * 
      arma::clamp(l, 0, arma::datum::inf));
  }
  
  return;
}



