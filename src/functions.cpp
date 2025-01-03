#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <cmath>
#include <iostream>
#include <stdexcept>  // std::invalid_argument
#include <vector>
#include "LBFGSB.h"

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using Eigen::VectorXd;


// [[Rcpp::export]]
void update_H_cpp(const arma::mat& X, const arma::mat& M, const arma::mat& W,
                       const arma::colvec& beta, arma::mat& H, 
                       const arma::colvec& y, const arma::colvec& delta, 
                       double alpha, bool WtX, const arma::uvec& ns) {
  arma::mat Wt = W.t();
  if(WtX){
    H.rows(ns) = H.rows(ns) % (Wt.rows(ns) * (M % X)) / (Wt.rows(ns) * (M % (W * H)));
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
    H = (H / (Wt * (M % (W * H))) ) %
      arma::clamp((Wt * (M % X)) + (alpha * arma::accu(M) / N) *
      l,0,arma::datum::inf);
    //H = H % (Wt * (M % X)) / (Wt * (M % (W * H)));
  }
  
  return;
}

// [[Rcpp::export]]
void update_W_cpp(const arma::mat& X, const arma::mat& Xt, const arma::mat& M,
                  const arma::mat& Mt, const arma::mat& H,
                  arma::mat& W, const arma::colvec& beta,
                  const arma::colvec& y, const arma::colvec& delta,
                  double alpha, bool WtX, int norm_type, const arma::uvec& ns,
                  double step, arma::mat& changeprev, double mo) {
  if(!WtX){
    W = W % ((M % X) * H.t()) / ((M % (W * H)) * H.t());
  }else{
    int N = X.n_cols;
    int P = X.n_rows;
    int s = arma::accu(M);
// 
//     // linear predictor
//     arma::vec lp = exp((Mt % Xt) * W * beta);
// 
//     // Indicator matrix
//     arma::mat y_matrix = arma::repmat(y, 1, N);
//     arma::mat Y = arma::conv_to<arma::mat>::from(y_matrix >= y_matrix.t());
//     arma::mat oneNP = arma::ones(N,P);
// 
//     // derivative of log likelihood
//     arma::mat LP = diagmat(lp);
//     arma::mat l = arma::kron(trans(delta) * ((Mt % Xt) - (trans(Y)*LP*(Mt % Xt))/(trans(Y)*LP*oneNP)),beta);
//     arma::mat Ht = H.t();
//     // arma::mat temp1 = (M % X) * Ht;
//     // arma::mat temp2 = (alpha * s / N) * l;
//     // arma::mat temp3 = temp1 + temp2;
//     // Rcout << temp1.rows(0,10) << "\n\n";
//     // Rcout << temp2.rows(0,10) << "\n\n";
//     // Rcout << temp3.rows(0,10) << "\n\n";
//     W.cols(ns) = (W.cols(ns) / (((M % (W*H)) * Ht.cols(ns))) ) %
//       arma::clamp(((M % X) * Ht.cols(ns)) + (alpha / N) * trans(l.rows(ns)),0,arma::datum::inf);
//       //arma::clamp(W % ((M % X) * Ht + (alpha * s / N) * trans(l)) / ((M % (W*H)) * Ht), 1e-10, arma::datum::inf);


    // linear predictor
    arma::vec lp = exp((Mt % Xt) * W * beta);

    // Indicator matrix
    arma::mat y_matrix = arma::repmat(y, 1, N);
    arma::mat Y = arma::conv_to<arma::mat>::from(y_matrix >= y_matrix.t());
    arma::mat oneNP = arma::ones(N,P);

    // derivative of log likelihood
    arma::mat LP = diagmat(lp);
    arma::mat l = arma::kron(trans(delta) * ((Mt % Xt) - (trans(Y)*LP*(Mt % Xt))/(trans(Y)*LP*oneNP)),beta);

    // compute gradient in matrix form
    arma::mat nmf = (M % (W*H - X)) * H.t() * 2.0; //* (2.0 / s);
    arma::mat like = alpha * 2 * l.t() / N;
    //Rcout << "test1\n";
    arma::mat gradient = nmf - like;
    // Rcout << "recon"<< nmf.rows(0,5) << "\n";
    // Rcout << "like"<< like.rows(0,5) << "\n";
    
    arma::mat change = step * gradient + mo * changeprev;

    W = W % arma::exp(-1 * change);
    
    changeprev = change;
  }


  // if(norm_type == 1){
  //   arma::colvec row_sums = sum(W, 1);
  //   W.each_col() /= row_sums;
  // }else if(norm_type == 2){
  //   arma::rowvec col_sums = sum(W, 0);
  //   W.each_row() /= col_sums;
  // }

  return;
}

//' @export
// [[Rcpp::export]]
double calc_surv_loss(const arma::mat& X, const arma::mat& M, 
                      const arma::mat& Wt,
                      const arma::vec& beta, const arma::vec& y,
                      const arma::vec& delta, bool WtX){
  int N = X.n_cols;
  arma::colvec lp;
  
  if(!WtX){
    //lp = H.t() * beta;
  }else{
    lp = trans(trans(beta) * Wt * (M % X));
  }
  
  //Rcout << "W:\n" << W.rows(0,5) << "\n";
  //Rcout << "XtW:\n" << trans(M % X) * W << "\n";
  //Rcout << "beta:\n" << beta << "\n";
  arma::mat y_matrix = arma::repmat(y, 1, N);
  arma::mat ind = arma::conv_to<arma::mat>::from(y_matrix >= y_matrix.t());
  
  return arma::accu(delta % (lp - arma::log(ind.t() * arma::exp(lp))));
  
}

// [[Rcpp::export]]
List calc_loss_cpp(const arma::mat& Xt, const arma::mat& X, 
                   const arma::mat& Mt, const arma::mat& M, 
                   const arma::mat& Wt, const arma::mat& Ht, const arma::mat& H,
                   const arma::vec& beta, double alpha, const arma::vec& y, 
                   const arma::vec& delta, double lambda, double eta, bool WtX, 
                   double gamma) {
  
  int s = arma::accu(M);
  double nmf_loss = arma::accu(arma::square(Mt % (Xt - Ht * Wt)));// / arma::accu(M);
  double surv_loss = calc_surv_loss(X, M, Wt, beta, y, delta, WtX);
  double penalty = lambda * ((1 - eta) * arma::accu(arma::square(beta)) + eta * arma::accu(arma::abs(beta)));
  double loss = (1-alpha) * (nmf_loss + gamma*arma::accu(Wt)) - alpha * s * (surv_loss - penalty);
  
  return List::create(
    Named("loss") = loss,
    Named("nmf_loss") = nmf_loss,
    Named("surv_loss") = surv_loss,
    Named("penalty") = penalty
  );
}

// class Hupdate
// {
// private:
//   const arma::mat& M;
//   const arma::mat& X;
//   const arma::vec& y;
//   const arma::vec& delta;
//   arma::mat& W;
//   arma::vec& beta;
//   double alpha;
//   double lambda;
//   double eta;
//   bool WtX;
// public:
//   
//   Hupdate(const arma::mat& M_, const arma::mat& X_, const arma::vec& y_,
//           const arma::vec& delta_, arma::mat& W_,
//           arma::vec& beta_, double alpha_, double lambda_,
//           double eta_, bool WtX_) : M(M_), X(X_), y(y_), delta(delta_), W(W_), beta(beta_),
//           alpha(alpha_), lambda(lambda_), eta(eta_), WtX(WtX_) {}
//   double operator()(const VectorXd& x, VectorXd& grad)
//   {
//     double fx = 0.0;
//     
//     
//     int s = arma::accu(M);
//     int k = W.n_cols;
//     int n = X.n_cols;
//     
//     //create H from x
//     //convert eigen vector x to std vector
//     std::vector<double> xstd(x.data(), x.data() + x.size());
//     //convert std vector to arma vector
//     arma::vec xarma = arma::conv_to< arma::colvec >::from(xstd);
//     //create H by stacking arma vector into columns
//     arma::mat H = arma::reshape(xarma,k,n);
//     int N = H.n_cols;
//     
//     // COMPUTE GRADIENT
//     // linear predictor
//     arma::vec lp = exp(H.t() * beta);
//     // Indicator matrix
//     arma::mat y_matrix = arma::repmat(y, 1, N);
//     arma::mat I = arma::conv_to<arma::mat>::from(y_matrix >= y_matrix.t());
//     // intermediate matrix
//     arma::mat temp = I.t() % arma::repmat(lp.t(),N,1) / arma::repmat(I.t() * lp, 1, N);
//     // derivative of log likelihood
//     arma::rowvec delta_t = delta.t();
//     arma::mat l = arma::kron(delta.t() - (delta.t() * temp),beta);
//     // compute gradient in matrix form
//     arma::mat nmf = W.t() * (M % (W*H - X)) * 2.0; //* (2.0 / s);
//     arma::mat like = alpha * 2 * l / N;
//     arma::mat gradient = nmf - like;
//     // convert to armadillo vector
//     arma::vec v = arma::vectorise(gradient);
//     // convert to standard vector
//     std::vector<double> v2 = arma::conv_to < std::vector<double> >::from(v);
//     // convert to eigen vectorXd
//     grad = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(v2.data(), v2.size());
//     
//     // Rcout << "nmf:\n" << nmf.submat(0,0,5-1,0) << "\n";
//     // Rcout << "like:\n" << like.submat(0,0,5-1,0) << "\n";
//     // Rcout << "grad:\n" << gradient.submat(0,0,5-1,0) << "\n";
//     //Rcout << "s:\n" << s << "\n";
//     
//     //COMPUTE FUNCTION VALUE
//     List loss = calc_loss_cpp(X,M,W,H,beta,alpha,y,delta,lambda,eta,WtX);
//     fx = loss["loss"];
//     return fx;
//   }
//   
//   void set_value(arma::mat& W_, arma::mat& beta_){
//     beta=beta_;
//     W=W_;
//   }
// };


class Wupdate
{
private:
  const arma::mat& Mt;
  const arma::mat& Xt;
  const arma::mat& M;
  const arma::mat& X;
  const arma::vec& y;
  const arma::vec& delta;
  arma::mat& H;
  arma::mat& Ht;
  arma::vec& beta;
  double alpha;
  double lambda;
  double eta;
  bool WtX;
  double gamma;
public:

  Wupdate(const arma::mat& Mt_, const arma::mat& M_, 
          const arma::mat& Xt_, const arma::mat& X_,
          const arma::vec& y_,
          const arma::vec& delta_, arma::mat& Ht_,
          arma::mat& H_,
          arma::vec& beta_, double alpha_, double lambda_,
          double eta_, bool WtX_, double gamma_) : Mt(Mt_), M(M_), Xt(Xt_), X(X_), y(y_), 
          delta(delta_), Ht(Ht_), H(H_), beta(beta_),
          alpha(alpha_), lambda(lambda_), eta(eta_), WtX(WtX_), gamma(gamma_) {}
  double operator()(const VectorXd& x, VectorXd& grad)
  {
    double fx = 0.0;


    int s = arma::accu(Mt);
    int k = Ht.n_cols;
    int p = Xt.n_cols;
    int n = Xt.n_rows;

    //create H from x
    //convert eigen vector x to std vector
    std::vector<double> xstd(x.data(), x.data() + x.size());
    //convert std vector to arma vector
    arma::vec xarma = arma::conv_to< arma::colvec >::from(xstd);
    //create W by stacking arma vector into columns
    arma::mat Wt = arma::reshape(xarma,k,p);

    // COMPUTE GRADIENT
    // linear predictor
    arma::vec lp = trans(exp(trans(beta) * Wt * (M % X)));

    // Indicator matrix
    arma::mat y_matrix = arma::repmat(y, 1, n);
    arma::mat Y = arma::conv_to<arma::mat>::from(y_matrix >= y_matrix.t());
    arma::mat oneNP = arma::ones(n,p);

    // derivative of log likelihood
    arma::mat LP = diagmat(lp);
    arma::mat l = arma::kron(trans(delta) * ((Mt % Xt) - (trans(Y)*LP*(Mt % Xt))/(trans(Y)*LP*oneNP)),beta);

    // compute gradient in matrix form
    arma::mat nmf =  (H * (Mt % (Ht*Wt - Xt)) * 2.0 + gamma) * (1 - alpha); //* (2.0 / s);
    arma::mat like = alpha * s * l;
    //Rcout << "test1\n";
    arma::mat gradient = nmf - like;
    //Rcout << "test2\n";
    // convert to armadillo vector
    arma::vec v = arma::vectorise(gradient);

    // convert to standard vector
    std::vector<double> v2 = arma::conv_to < std::vector<double> >::from(v);
    // convert to eigen vectorXd
    grad = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(v2.data(), v2.size());

    // Rcout << "nmf:\n" << nmf.submat(0,0,5-1,0) << "\n";
    // Rcout << "like:\n" << like.submat(0,0,5-1,0) << "\n";
    // Rcout << "grad:\n" << gradient.submat(0,0,5-1,0) << "\n";
    //Rcout << "s:\n" << s << "\n";

    //COMPUTE FUNCTION VALUE
    List loss = calc_loss_cpp(Xt,X,Mt,M,Wt,Ht,H,beta,alpha,y,delta,lambda,eta,WtX,gamma);
    fx = loss["loss"];
    return fx;
  }

  void set_value(arma::mat& beta_, arma::mat& H_){
    beta=beta_;
    H=H_;
    Ht=trans(H);
  }
};

// Everything for updating beta below

// double MCP(double z, double l1, double l2, double gamma, double v) {
//   double s = 0;
//   
//   // Determine the sign of z
//   if (z > 0) 
//     s = 1;
//   else if (z < 0) 
//     s = -1;
//   
//   // Apply MCP penalty based on the value of z
//   if (std::abs(z) <= l1) 
//     return 0; // No penalty
//   else if (std::abs(z) <= gamma * l1 * (1 + l2)) 
//     return s * (std::abs(z) - l1) / (v * (1 + l2 - 1 / gamma)); // MCP penalty
//   else 
//     return z / (v * (1 + l2));
// }

// double SCAD(double z, double l1, double l2, double gamma, double v) {
//   double s = 0;
//   
//   // Determine the sign of z
//   if (z > 0) 
//     s = 1;
//   else if (z < 0) 
//     s = -1;
//   
//   // Apply SCAD penalty based on the value of z
//   if (std::abs(z) <= l1) 
//     return 0; // No penalty
//   else if (std::abs(z) <= (l1 * (1 + l2) + l1)) 
//     return s * (std::abs(z) - l1) / (v * (1 + l2)); // SCAD penalty
//   else if (std::abs(z) <= gamma * l1 * (1 + l2)) 
//     return s * (std::abs(z) - gamma * l1 / (gamma - 1)) / (v * (1 - 1 / (gamma - 1) + l2)); // SCAD penalty
//   else 
//     return z / (v * (1 + l2)); // No penalty
// }

double lasso(double z, double l1, double l2, double v) {
  double s = 0;
  
  // Determine the sign of z
  if (z > 0) 
    s = 1;
  else if (z < 0) 
    s = -1;
  
  // Apply lasso penalty based on the value of z
  if (std::abs(z) <= l1) 
    return 0; // No penalty
  else 
    return s * (std::abs(z) - l1) / (v * (1 + l2)); // Lasso penalty
}

// [[Rcpp::export]]
arma::mat cdfit_cox_dh(const arma::mat& X, const arma::vec& d, String penalty, 
                       const arma::vec& lambda, double eps, int max_iter, double gamma,
                       const arma::vec& m, double alpha, int dfmax, bool user, bool warn) {
  
  int n = X.n_rows;
  int p = X.n_cols;
  int L = lambda.n_elem;
  
  arma::mat b = arma::zeros<arma::mat>(p,L);
  arma::vec Loss = arma::zeros<arma::vec>(L);
  arma::ivec iter = arma::zeros<arma::ivec>(L);
  arma::vec Eta = arma::zeros<arma::vec>(L*n);
  arma::vec a = arma::zeros<arma::vec>(p);
  arma::vec haz = arma::zeros<arma::vec>(n);
  arma::vec rsk = arma::zeros<arma::vec>(n);
  arma::vec r = arma::zeros<arma::vec>(n);
  arma::vec h = arma::zeros<arma::vec>(n);
  arma::ivec e = arma::zeros<arma::ivec>(p);
  arma::vec eta = arma::zeros<arma::vec>(n);
  
  // stopped here, still need to double check the zero initialization above
  
  double xwr, xwx, u, v, l1, l2, shift, si, s, nullDev;
  int lstart;
  
  rsk[n-1] = 1;
  for (int i = n - 2; i >= 0; i--) rsk[i] = rsk[i + 1] + 1;
  nullDev = 0;
  for (int i = 0; i < n; i++) nullDev -= d[i] * std::log(rsk[i]);
  lstart = user ? 0 : 1;
  if (!user) Loss[0] = nullDev;
  
  int tot_iter = 0;
  for (int l = lstart; l < L; l++) {
    Rcpp::checkUserInterrupt();
    if (l != 0) {
      a = b.col(l-1);
      int nv = arma::accu(a != 0);
      if (nv > dfmax || tot_iter == max_iter) {
        iter.subvec(l, L - 1).fill(NA_INTEGER);
        break;
      }
    }
    
    while (tot_iter < max_iter) {
      while(tot_iter < max_iter) {
        iter[l]++;
        tot_iter++;
        Loss[l] = 0;
        double maxChange = 0;
        
        haz = arma::exp(eta);
        rsk[n-1] = haz[n-1];
        for (int i = n - 2; i >= 0; i--) rsk[i] = rsk[i + 1] + haz[i];
        for (int i = 0; i < n; i++) Loss[l] += d[i] * eta[i] - d[i] * std::log(rsk[i]);
        
        h[0] = d[0] / rsk[0];
        for (int i = 1; i < n; i++) h[i] = h[i - 1] + d[i] / rsk[i];
        for (int i = 0; i < n; i++) {
          h[i] *= haz[i];
          s = d[i] - h[i];
          r[i] = h[i] == 0 ? 0 : s / h[i];
        }
        
        if (Loss[l] / nullDev < .01) {
          if (warn) Rcpp::warning("Model saturated; exiting...");
          iter.subvec(l, L - 1).fill(NA_INTEGER);
          tot_iter = max_iter;
          break;
        }
        
        for (int j = 0; j < p; j++) {
          if (e[j]) {
            xwr = arma::accu(X.col(j) % r % h);
            xwx = arma::accu(h % arma::square(X.col(j)));
            u = xwr / n + (xwx / n) * a[j];
            v = xwx / n;
            
            l1 = lambda[l] * m[j] * alpha;
            l2 = lambda[l] * m[j] * (1 - alpha);
            //if (penalty == "MCP") b(j,l) = MCP(u, l1, l2, gamma, v);
            //if (penalty == "SCAD") b(j,l) = SCAD(u, l1, l2, gamma, v);
            if (penalty == "lasso") b(j,l) = lasso(u, l1, l2, v);
            
            shift = b(j,l) - a[j];
            if (shift != 0) {
              for (int i = 0; i < n; i++) {
                si = shift * X(i, j);
                r[i] -= si;
                eta[i] += si;
              }
              maxChange = std::max(maxChange, std::abs(shift) * std::sqrt(v));
            }
          }
        }
        
        a = b.col(l);
        if (maxChange < eps) break;
      }
      
      int violations = 0;
      for (int j = 0; j < p; j++) {
        if (!e[j]) {
          xwr = arma::accu(X.col(j) % r % h) / n;
          l1 = lambda[l] * m[j] * alpha;
          if (std::abs(xwr) > l1) {
            e[j] = 1;
            violations++;
          }
        }
      }
      if (violations == 0) {
        Eta.subvec(l * n, (l + 1) * n - 1) = eta;
        break;
      }
    }
  }
  
  return b;
}

// [[Rcpp::export]]
arma::vec cdfit_cox_dh_one_lambda(const arma::mat& X, const arma::vec& d, String penalty,
                  double lambda, double eps, int max_iter,
                  const arma::vec& m, double alpha) { //double gamma

  int n = X.n_rows;
  int p = X.n_cols;
  double Loss;
  arma::vec b = arma::zeros<arma::vec>(p);

  arma::vec a = arma::zeros<arma::vec>(p);
  arma::vec haz = arma::zeros<arma::vec>(n);
  arma::vec rsk = arma::zeros<arma::vec>(n);
  arma::vec r = arma::zeros<arma::vec>(n);
  arma::vec h = arma::zeros<arma::vec>(n);
  arma::vec eta = arma::zeros<arma::vec>(n);

  // stopped here, still need to double check the zero initialization above

  double xwr, xwx, u, v, l1, l2, shift, si, s, nullDev;

  rsk[n-1] = 1;
  for (int i = n - 2; i >= 0; i--) rsk[i] = rsk[i + 1] + 1;
  nullDev = 0;
  for (int i = 0; i < n; i++) nullDev -= d[i] * std::log(rsk[i]);

  int tot_iter = 0;

  while (tot_iter < max_iter) {
    tot_iter++;
    Loss = 0;
    double maxChange = 0;

    haz = arma::exp(eta);
    rsk[n-1] = haz[n-1];
    for (int i = n - 2; i >= 0; i--) rsk[i] = rsk[i + 1] + haz[i];
    for (int i = 0; i < n; i++) Loss += d[i] * eta[i] - d[i] * std::log(rsk[i]);

    h[0] = d[0] / rsk[0];
    for (int i = 1; i < n; i++) h[i] = h[i - 1] + d[i] / rsk[i];
    for (int i = 0; i < n; i++) {
      h[i] *= haz[i];
      s = d[i] - h[i];
      r[i] = h[i] == 0 ? 0 : s / h[i];
    }


    for (int j = 0; j < p; j++) {
      xwr = arma::accu(X.col(j) % r % h);
      xwx = arma::accu(h % arma::square(X.col(j)));
      u = xwr / n + (xwx / n) * a[j];
      v = xwx / n;

      l1 = lambda * m[j] * alpha;
      l2 = lambda * m[j] * (1 - alpha);
      // if (penalty == "MCP") b[j] = MCP(u, l1, l2, gamma, v);
      // if (penalty == "SCAD") b[j] = SCAD(u, l1, l2, gamma, v);
      if (penalty == "lasso") b[j] = lasso(u, l1, l2, v);

      shift = b[j] - a[j];
      if (shift != 0) {
        for (int i = 0; i < n; i++) {
          si = shift * X(i, j);
          r[i] -= si;
          eta[i] += si;
        }
        maxChange = std::max(maxChange, std::abs(shift) * std::sqrt(v));
      }
    }
    a = b;
    if (maxChange < eps) break;
  }

  return b;
}


// [[Rcpp::export]]
arma::vec cdfit_cox_dh_one_lambda_it(const arma::mat& X, const arma::vec& d, String penalty,
                                     double lambda, const arma::vec& a,
                                     const arma::vec& m, double alpha) { //double gamma
  
  int n = X.n_rows;
  int p = X.n_cols;
  double Loss = 0;
  arma::vec b = arma::zeros<arma::vec>(p);
  
  arma::vec haz = arma::zeros<arma::vec>(n);
  arma::vec rsk = arma::zeros<arma::vec>(n);
  arma::vec r = arma::zeros<arma::vec>(n);
  arma::vec h = arma::zeros<arma::vec>(n);
  arma::vec eta = X * a;
  
  double xwr, xwx, u, v, l1, l2, shift, si, s, nullDev;
  
  rsk[n-1] = 1;
  for (int i = n - 2; i >= 0; i--) rsk[i] = rsk[i + 1] + 1;
  nullDev = 0;
  for (int i = 0; i < n; i++) nullDev -= d[i] * std::log(rsk[i]);
  
  haz = arma::exp(eta);
  rsk[n-1] = haz[n-1];
  for (int i = n - 2; i >= 0; i--) rsk[i] = rsk[i + 1] + haz[i];
  for (int i = 0; i < n; i++) Loss += d[i] * eta[i] - d[i] * std::log(rsk[i]);
  
  h[0] = d[0] / rsk[0];
  for (int i = 1; i < n; i++) h[i] = h[i - 1] + d[i] / rsk[i];
  for (int i = 0; i < n; i++) {
    h[i] *= haz[i];
    s = d[i] - h[i];
    r[i] = h[i] == 0 ? 0 : s / h[i];
  }
  
  
  for (int j = 0; j < p; j++) {
    xwr = arma::accu(X.col(j) % r % h);
    xwx = arma::accu(h % arma::square(X.col(j)));
    u = xwr / n + (xwx / n) * a[j];
    v = xwx / n;
    
    l1 = lambda * m[j] * alpha;
    l2 = lambda * m[j] * (1 - alpha);
    // if (penalty == "MCP") b[j] = MCP(u, l1, l2, gamma, v);
    // if (penalty == "SCAD") b[j] = SCAD(u, l1, l2, gamma, v);
    if (penalty == "lasso") b[j] = lasso(u, l1, l2, v);
    
    shift = b[j] - a[j];
    if (shift != 0) {
      for (int i = 0; i < n; i++) {
        si = shift * X(i, j);
        r[i] -= si;
        eta[i] += si;
      }
    }
  }
  
  return b;
}



// [[Rcpp::export]]
arma::vec update_beta_cpp(const arma::mat& X, const arma::mat& y, String penalty,
                          double alpha, double lambda, const arma::vec& beta0){


  // Order y by time
  arma::uvec tOrder = arma::sort_index(y.col(0));
  arma::vec yy = arma::conv_to<arma::vec>::from(y.col(0)).elem(tOrder);
  arma::vec Delta = arma::conv_to<arma::vec>::from(y.col(1)).elem(tOrder);
  arma::mat XX = X.rows(tOrder);
  // Standardize X
  arma::rowvec meanX = arma::mean(XX, 0);
  arma::rowvec sdX = arma::stddev(XX, 1, 0); // using unbiased estimator
  arma::uvec ns = arma::find(sdX > .000001);
  XX = XX.cols(ns);
  meanX = arma::mean(XX, 0);
  sdX = arma::stddev(XX, 1, 0); 
  //Rcout << sdX << "\n";
  XX.each_row() -= meanX;
  XX.each_row() /= sdX;
  //Rcout << XX.rows(0,4) << "\n";

  int p = XX.n_cols;

  arma::vec penalty_factor = arma::ones<arma::vec>(p);
  //Rcout << "test1\n";
  // perform coordinate descent
  arma::vec b = cdfit_cox_dh_one_lambda_it(XX, Delta, penalty, lambda,
                                           beta0.elem(ns), penalty_factor, alpha);
  //Rcout << "test2\n";
  // Unstandardize coefficients
  arma::mat beta = arma::zeros<arma::vec>(X.n_cols);
  arma::vec bb = b / sdX.t();
  beta.elem(ns) = bb;
  //Rcout << "test3\n";
  return beta;
}

//' @export
// [[Rcpp::export]]
void standardize(arma::mat& W, arma::mat& H, arma::colvec& beta, int norm_type,
                 bool WtX, arma::uvec ns){
  
  
  // if(WtX){
    arma::mat HH = H.rows(ns);
    arma::rowvec col_sum = sum(HH, 0);
    HH.each_row() /= col_sum;
    H.rows(ns) = HH;
  // }else{
    // arma::mat WW = W.cols(ns);
    // arma::rowvec col_sum = sum(WW, 0);
    // WW.each_row() /= col_sum;
    // W.cols(ns) = WW;
  // }
  
  
    // arma::rowvec col_sums = sum(W, 0);
    // W.each_row() /= col_sums;
    // H.each_col() %= col_sums.t();
    // beta /= col_sums.t();
    
  // arma::colvec row_sums = sum(W,1);
  // W.each_col() /= row_sums;
  // H.each_row() 
  
  return;
}

void sparsity(arma::mat& W, int num_genes){

  int p = W.n_rows;
  int k = W.n_cols;
  arma::vec q = {1 - 1.0*num_genes/p};
  Rcout << q << "\n";
  arma::rowvec thresh = arma::conv_to<arma::rowvec>::from(quantile(W,q));
  Rcout << thresh << "\n";
  // Rcout << arma::sum(W.each_row() >= thresh,0) << "\n";
  
  // for(int j=0; j<=(k-1); j++){
  //   for(int i=0; i<=(p-1); i++){
  //     W(i,j) = W(i,j) < thresh(j);
  //   }
  // }
  W.each_row([thresh](arma::rowvec& row){
    row.elem(arma::find(row < thresh)).zeros();
  });

  Rcout << arma::sum(W != 0, 0) << "\n";

   return;
}

//' @export
// [[Rcpp::export]]
List optimize_loss_cpp(const arma::mat& X, const arma::mat& M,
                            const arma::mat& H0, const arma::mat& W0,
                            const arma::colvec& beta0, const arma::colvec& y,
                            const arma::colvec& delta, double alpha,
                            double lambda, double eta, double tol,
                            int maxit, bool verbose, bool WtX, int norm_type,
                            String penalty, bool init, double step, double mo,
                            bool BFGS, int num_genes, double gamma){
  arma::mat H = H0;
  arma::mat Ht = trans(H);
  arma::mat W = W0;
  arma::mat Xt = trans(X);
  arma::mat Mt = trans(M);
  arma::mat Wt = trans(W);
  
  arma::colvec beta = beta0;
  
  
  
  int N = H.n_cols;
  int P = X.n_rows;
  int k = H.n_rows;
  
  arma::uvec ns = arma::conv_to<arma::uvec>::from(arma::linspace(0,k-1,k));
  
  double loss = 0.000001;
  double eps = 1;
  int it = 1;
  double loss_prev;
  List b;
  List l;
  arma::mat s = arma::join_horiz(y,delta);
  arma::mat XtW;
  arma::mat changeprev = arma::zeros<arma::mat>(P,k);
  
  LBFGSpp::LBFGSBParam<double> param;
  // param.epsilon = 1e-5;
  // param.max_iterations = 100;
  // param.min_step = 1e-30;
  double fx;
  // Create solver and function object
  LBFGSpp::LBFGSBSolver<double> solver(param);
  // Declare function object Hupdate
  Wupdate fun(Mt,M,Xt,X,y,delta,Ht,H,beta,alpha,lambda,eta,WtX, gamma);

  // bounds for constrained optimization
  VectorXd lb = VectorXd::Constant(P*k, 0.0);
  VectorXd ub = VectorXd::Constant(P*k, std::numeric_limits<double>::infinity());

  // declare intermediate vectors
  arma::vec xarma;
  std::vector<double> xstd;
  VectorXd x;
  std::vector<double> xstd2;
  arma::vec xarma2;
  arma::vec lossit = arma::zeros<arma::vec>(maxit);
  arma::vec slossit = arma::zeros<arma::vec>(maxit);
  arma::vec nlossit = arma::zeros<arma::vec>(maxit);
  arma::vec plossit = arma::zeros<arma::vec>(maxit);
  
  
  while(eps > tol && it <= maxit){
    loss_prev = loss;// fun.set_value(W,beta);
    
    
    update_H_cpp(X,M,W,beta,H,y,delta,alpha,WtX,ns);
    standardize(W,H,beta,norm_type,WtX,ns);
    
    XtW = trans(M % X) * W;
    
    if(WtX){
      beta = update_beta_cpp(XtW, s, penalty,eta,lambda,beta);
    }else{
      //beta = update_beta_cpp(H.t(),s,penalty,eta,lambda,beta);
    }
    
    
    if(BFGS){
      fun.set_value(beta,H);
      
      // convert H to arma::vec
      xarma = arma::vectorise(Wt);
      // convert to standard vector
      xstd = arma::conv_to < std::vector<double> >::from(xarma);
      // convert to eigen vectorXd
      x = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(xstd.data(), xstd.size());
      
      int niter = solver.minimize(fun, x, fx, lb, ub);
      //now convert x back to H
      //convert eigen vector x to std vector
      xstd2.assign(x.data(), x.data() + x.size());
      // //convert std vector to arma vector
      xarma2 = arma::conv_to< arma::colvec >::from(xstd2);
      // //create H by stacking arma vector into columns
      Wt = arma::reshape(xarma2,k,P);
      
      W = trans(Wt);
      //Rcout << W.rows(0,4) << "\n";
      //sparsity(W,num_genes);
    }else{
      update_W_cpp(X,Xt,M,Mt,H,W,beta,y,delta,alpha,WtX,norm_type,ns,step,changeprev,mo);
    }
    
    
    // if(it==93){
    //   Rcout << "H:\n" << H << "\n";
    // }
    // if(it==93){
    //   arma::mat WH = W * H;
    //   Rcout << "WH3314nan\n" << WH.submat(3314,0,3314,0) << "\n";
    // }
    // if(it==93){
    //   Rcout << "W3314\n" << W.rows(3314,3314) << "\n";
    // }
    //update_W_cpp(X,Xt,M,Mt,H,W,beta,y,delta,alpha,WtX,norm_type,ns);
    // 
    //Rcout << "Xt:\n" << Xt.cols(0,15) << "\n";
    //find the active set
    
    // Rcout << "XtW:\n" << XtW.rows(0,6) << "\n";
    // arma::rowvec sdX = arma::stddev(XtW, 1, 0); // using unbiased estimator
    // ns = arma::find(sdX > .000001);
    // Rcout << "sdX\n" << sdX << "\n";
    // Rcout << "ns:\n" << ns << "\n";
        // if(it==725){
    //   Rcout << "W:\n" << W.cols(0,0) << "\n";
    // }
    
    
    
    
    

    //Rcout << "test 1" << "\n";
   
    
    
    
    // if(alpha>0){

    // }

    // fun.set_value(W,beta);
    // 
    // //update_H_cpp(X,M,W,beta,H,y,delta,alpha,WtX);
    // // convert H to arma::vec
    // xarma = arma::vectorise(H);
    // // convert to standard vector
    // xstd = arma::conv_to < std::vector<double> >::from(xarma);
    // // convert to eigen vectorXd
    // x = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(xstd.data(), xstd.size());
    // 
    // int niter = solver.minimize(fun, x, fx, lb, ub);
    // //now convert x back to H
    // //convert eigen vector x to std vector
    // xstd2.assign(x.data(), x.data() + x.size());
    // // //convert std vector to arma vector
    // xarma2 = arma::conv_to< arma::colvec >::from(xstd2);
    // // //create H by stacking arma vector into columns
    // H = arma::reshape(xarma2,k,N);
    // 
    // 
    // // standardize
    
    //Rcout << "Hstd:\n" << H.cols(0,4) << "\n";
    // 
    // arma::mat lptemp = H.t() * beta;
    
    

    l = calc_loss_cpp(Xt, X, Mt, M, W.t(), H.t(), H, beta, 
                      alpha, y, delta, lambda, eta, WtX, gamma);
    loss = l["loss"];
    lossit[it-1] = loss;
    
    double survloss = l["surv_loss"];
    slossit[it-1] = survloss;
    double nmfloss = l["nmf_loss"];
    nlossit[it-1] = nmfloss;
    double penloss = l["penalty"];
    plossit[it-1] = penloss;
    
    // Rcout << "loss\n" << loss << "\n";
    // Rcout << "surv loss\n" << survloss*alpha << "\n";
    // Rcout << "nmf loss\n" << nmfloss << "\n";
    // Rcout << "penalty\n" << penloss << "\n";
    // 
    // Rcout << "W\n" << W.rows(0,4) << "\n";
    // Rcout << "H\n" << H.cols(0,4) << "\n";
    // Rcout << "beta\n" << beta << "\n";
    // Rcout << "lp\n" << lptemp.rows(0,4) << "\n";

    eps = std::abs(loss - loss_prev)/loss_prev;

    
    if(verbose){
      Rprintf("iter: %d eps: %.8f loss: %.8f\n",it,eps,loss);
    }
    if(it == maxit && !init && verbose){
      warning("coxNMF failed to converge");
    }
    it = it + 1;

  }

  List L = List::create(
    Named("W") = W,
    Named("H") = H,
    Named("beta") = beta,
    Named("loss") = l,
    Named("lossit") = lossit,
    Named("slossit") = slossit,
    Named("nlossit") = nlossit,
    Named("plossit") = plossit,
    Named("iter") = it);
  return L;
}
