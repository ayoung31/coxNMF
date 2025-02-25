#include <RcppArmadillo.h>


// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;


// [[Rcpp::export]]
void update_H_cpp(const arma::mat& X, const arma::mat& M, 
                  const arma::colvec& y, const arma::colvec& delta, 
                  const arma::mat& W, arma::mat& H) {
  
  arma::mat Wt = W.t();
  H = H % (Wt * (M % X)) / (Wt * (M % (W * H)));
  
  return;
}

// [[Rcpp::export]]
void update_W_cpp(const arma::mat& X, const arma::mat& M, 
                  const arma::colvec& y, const arma::colvec& delta,
                  arma::mat& W, const arma::mat& H, const arma::colvec& beta, 
                  double alpha) {

  int N = X.n_cols;
  int P = X.n_rows;
  int s = arma::accu(M);
  
  //necessary transposed matrices
  arma::mat Ht = H.t();
  arma::mat Xt = X.t();
  arma::mat Mt = M.t();

  // linear predictor
  arma::vec lp = exp((Mt % Xt) * W * beta);

  // Indicator matrix
  arma::mat y_matrix = arma::repmat(y, 1, N);
  arma::mat Y = arma::conv_to<arma::mat>::from(y_matrix >= y_matrix.t());
  arma::mat oneNP = arma::ones(N,P);

  // derivative of log likelihood
  arma::mat LP = diagmat(lp);
  arma::mat l = arma::kron(trans(delta) * ((Mt % Xt) - (trans(Y)*LP*(Mt % Xt))/(trans(Y)*LP*oneNP)),beta);
  
  arma::mat inside = arma::clamp(((M % X) * Ht) + (alpha * s / (2 * (1-alpha))) * trans(l),0,arma::datum::inf);
  arma::mat denom = (M % (W*H)) * Ht;
  W = (W / denom) % inside;
    
  W.elem( find_nonfinite(W) ).zeros();
  
  return;
}

//' @export
// [[Rcpp::export]]
double calc_surv_loss(const arma::mat& X, const arma::mat& M, 
                      const arma::vec& y, const arma::vec& delta,
                      const arma::mat& W, const arma::vec& beta){
  int N = X.n_cols;
  // int N = H.n_cols;
  arma::colvec lp;

  lp = trans(M % X) * W * beta;
  arma::mat y_matrix = arma::repmat(y, 1, N);
  arma::mat ind = arma::conv_to<arma::mat>::from(y_matrix >= y_matrix.t());
  
  return arma::accu(delta % (lp - arma::log(ind.t() * arma::exp(lp))));
  
}

//' @export
// [[Rcpp::export]]
List calc_loss_cpp(const arma::mat& X, const arma::mat& M, 
                   const arma::vec& y, const arma::vec& delta, 
                   const arma::mat& W, const arma::mat& H, const arma::vec& beta, 
                   double alpha, double lambda, double eta) {
  
  
  double nmf_loss = arma::accu(arma::square(M % (X - W * H)));// / arma::accu(M);
  double surv_loss = calc_surv_loss(X, M, y, delta, W, beta);
  double penalty = lambda * ((1 - eta) * arma::accu(arma::square(beta)) / 2 + eta * arma::accu(arma::abs(beta)));
  double loss = (1-alpha)*nmf_loss - alpha * arma::accu(M)*(surv_loss - penalty);
  
  return List::create(
    Named("loss") = loss,
    Named("nmf_loss") = nmf_loss,
    Named("surv_loss") = surv_loss,
    Named("penalty") = penalty
  );
}

// Everything for updating beta below
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
arma::mat cdfit_cox_dh(const arma::mat& X, const arma::vec& d,
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
            b(j,l) = lasso(u, l1, l2, v);
            
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
arma::vec cdfit_cox_dh_one_lambda(const arma::mat& X, const arma::vec& d, 
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
      b[j] = lasso(u, l1, l2, v);

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
arma::vec cdfit_cox_dh_one_lambda_it(const arma::mat& X, const arma::vec& d,
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
    b[j] = lasso(u, l1, l2, v);
    
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
arma::vec update_beta_cpp(const arma::mat& X, const arma::mat& y,
                          double alpha, double lambda, arma::vec beta0){

  // Order y by time
  arma::uvec tOrder = arma::sort_index(y.col(0));
  arma::vec yy = arma::conv_to<arma::vec>::from(y.col(0)).elem(tOrder);
  arma::vec Delta = arma::conv_to<arma::vec>::from(y.col(1)).elem(tOrder);
  arma::mat XX = X.rows(tOrder);
  // Standardize X
  arma::rowvec meanX = arma::mean(XX, 0);
  arma::rowvec sdX = arma::stddev(XX, 1, 0); // using unbiased estimator
  //Rcout << sdX << "\n";
  XX.each_row() -= meanX;
  XX.each_row() /= sdX;
  //Rcout << XX.rows(0,4) << "\n";
  arma::uvec ns = arma::find(sdX > .000001);
  XX = XX.cols(ns);
  int p = XX.n_cols;

  arma::vec penalty_factor = arma::ones<arma::vec>(p);
  penalty_factor = penalty_factor.elem(ns);
  // perform coordinate descent
  arma::vec b = cdfit_cox_dh_one_lambda_it(XX, Delta, lambda,
                                           beta0, penalty_factor, alpha);

  // Unstandardize coefficients
  arma::vec beta = arma::zeros<arma::vec>(X.n_cols);
  arma::vec bb = b / sdX.t();
  beta.elem(ns) = bb;

  return beta;
}

//' @export
// [[Rcpp::export]]
void standardize(arma::mat& W, arma::mat& H, arma::colvec& beta){
  
    arma::colvec row_sum = sum(H, 1);
    H.each_col() /= row_sum;

  
  return;
}

//' @export
// [[Rcpp::export]]
List optimize_loss_cpp(const arma::mat& X, const arma::mat& M,
                       const arma::colvec& y, const arma::colvec& delta,
                       const arma::mat& W0, const arma::mat& H0, 
                       const arma::colvec& beta0,  
                       double alpha, double lambda, double eta, 
                       double tol, int maxit, bool verbose, bool init){
  arma::mat H = H0;
  arma::mat W = W0;
  arma::colvec beta = beta0;
  
  int N = H.n_cols;
  int P = X.n_rows;
  int k = H.n_rows;
  
  double loss = 0.000001;
  double eps = 1;
  int it = -1;
  double loss_prev;
  List l;
  arma::mat s = arma::join_horiz(y,delta);

  arma::vec lossit = arma::zeros<arma::vec>(maxit);
  
  while(eps > tol && it < maxit){
    it = it + 1;
    
    loss_prev = loss;
    
    update_W_cpp(X, M, y, delta, W, H, beta, alpha);
    //Rcout << "W:\n" << W.rows(0,4) << "\n";

    beta = update_beta_cpp(trans(M % X) * W, s, eta, lambda, beta);

    //Rcout << "beta:\n" << beta << "\n";


    update_H_cpp(X, M, y, delta, W, H);
    standardize(W,H,beta);
    
    l = calc_loss_cpp(X, M, y, delta, W, H, beta, alpha, lambda, eta);

    loss = l["loss"];
    //Rcout << "loss: " << loss << "\n";
    
    double survloss = l["surv_loss"];
    //Rcout << "surv loss: " << survloss << "\n";
    double nmfloss = l["nmf_loss"];
    //Rcout << "nmf loss: " << nmfloss << "\n";
    double penloss = l["penalty"];
    
    // Rcout << "loss\n" << loss << "\n";
    // Rcout << "surv loss\n" << survloss << "\n";
    // Rcout << "nmf loss\n" << nmfloss << "\n";
    // Rcout << "penalty\n" << penloss << "\n";
    // 
    // if(it>2000){
    //   Rcout << "W\n" << W.rows(0,8) << "\n";
    //   Rcout << "H\n" << H.cols(0,8) << "\n";
    // }
    
    // 
    // Rcout << "beta\n" << beta << "\n";
    // Rcout << "lp\n" << lptemp.rows(0,4) << "\n";

    eps = std::abs(loss - loss_prev)/loss_prev;
    
    lossit[it] = loss;

    
    if(verbose){
      Rprintf("iter: %d eps: %.8f loss: %.8f\n",it,eps,loss);
    }
    if(it == maxit && !init){
      warning("coxNMF failed to converge");
    }

  }
  
  List L = List::create(
    Named("W") = W,
    Named("H") = H,
    Named("beta") = beta,
    Named("loss") = l,
    Named("iter") = it,
    Named("lossit") = lossit);
  return L;
}
