#include <RcppArmadillo.h>


// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;


// [[Rcpp::export]]
void update_H_cpp(const arma::mat& X, const arma::mat& M, 
                  const arma::colvec& y, const arma::colvec& delta, 
                  const arma::mat& W, arma::mat& H, double alpha,
                  double std_nmf, double lambdaH) {
  
  
  
  arma::mat Wt = W.t();
  arma::mat denom = Wt * (M % (W * H)) + (std_nmf*lambdaH / (1-alpha)) * H;
  //Rcout << "denom: \n" << denom << "\n";
  arma::mat num = H % (Wt * (M % X));
  //Rcout << "num: \n" << num << "\n";
  
  arma::uvec indices = arma::find(num>0 && denom>0);
  
  H.elem(indices) = num.elem(indices) / denom.elem(indices);
  
  return;
}

// [[Rcpp::export]]
void update_W_cpp(const arma::mat& X, const arma::mat& M, 
                  const arma::colvec& y, const arma::colvec& delta,
                  arma::mat& W, const arma::mat& H, const arma::colvec& beta, 
                  double alpha, double std_nmf, double std_surv, double lambdaW) {

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
  
  arma::mat inside = arma::clamp(((M % X) * Ht) + (alpha * std_nmf / (2 * (1-alpha) * std_surv)) * trans(l),0,arma::datum::inf);
  arma::mat denom = (M % (W*H)) * Ht + (lambdaW * std_nmf / (1-alpha)) * W;
  
  //only update nonzero elements of W
  arma::uvec indices = arma::find(W);
  
  W.elem(indices) = (W.elem(indices) / denom.elem(indices)) % inside.elem(indices);
  
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
                   double alpha, double lambda, double eta, 
                   double std_nmf, double std_surv,
                   double lambdaW, double lambdaH) {
  
  
  double nmf_loss = arma::accu(arma::square(M % (X - W * H)))/std_nmf;
  
  double penalty_beta = lambda * ((1 - eta) * arma::accu(arma::square(beta)) / 2 + 
                                  eta * arma::accu(arma::abs(beta))); 
  double surv_loss = calc_surv_loss(X, M, y, delta, W, beta) /std_surv + penalty_beta;
  double penalty_W = lambdaW * arma::accu(arma::square(W));
  double penalty_H = lambdaH * arma::accu(arma::square(H));
  double penalty = penalty_W + penalty_H;
  double loss = (1-alpha)*nmf_loss - alpha * surv_loss + penalty;
  
  return List::create(
    Named("loss") = loss,
    Named("nmf_loss") = nmf_loss,
    Named("surv_loss") = surv_loss,
    Named("penalty") = penalty,
    Named("penalty_beta") = penalty_beta,
    Named("penalty_W") = penalty_W,
    Named("penalty_H") = penalty_H
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
                                     const arma::vec& m, double alpha, int it,
                                     bool& flag_nan) { //double gamma
  
  int n = X.n_rows;
  int p = X.n_cols;
  double Loss = 0;
  arma::vec b = arma::zeros<arma::vec>(p);
  
  arma::vec haz = arma::zeros<arma::vec>(n);
  arma::vec rsk = arma::zeros<arma::vec>(n);
  arma::vec r = arma::zeros<arma::vec>(n);
  arma::vec h = arma::zeros<arma::vec>(n);
  arma::vec eta = X * a;
  haz = arma::exp(eta);
  
  //Rcout << "eta:\n" << eta.t() << "\n";
  //Rcout << "haz:\n" << haz.t() << "\n";
  
  if(!haz.is_finite()){
    flag_nan = TRUE;
    return(a);
  }
  
  
  double xwr, xwx, u, v, l1, l2, shift, si, s, nullDev;
  
  rsk[n-1] = 1;
  for (int i = n - 2; i >= 0; i--) rsk[i] = rsk[i + 1] + 1;
  
  nullDev = 0;
  for (int i = 0; i < n; i++) nullDev -= d[i] * std::log(rsk[i]);

  
  
  rsk[n-1] = haz[n-1];
  for (int i = n - 2; i >= 0; i--) rsk[i] = rsk[i + 1] + haz[i];

  //Rcout << "rsk:\n" << rsk.t() << "\n";
    
  for (int i = 0; i < n; i++) Loss += d[i] * eta[i] - d[i] * std::log(rsk[i]);
  
  //Rcout << "loss: " << Loss << "\n";

  h[0] = d[0] / rsk[0];
  for (int i = 1; i < n; i++) h[i] = h[i - 1] + d[i] / rsk[i];

  //Rcout << "h:\n" << h.t() << "\n";
    
  for (int i = 0; i < n; i++) {
    h[i] *= haz[i];
    s = d[i] - h[i];
    r[i] = h[i] == 0 ? 0 : s / h[i];
  }

  //Rcout << "h:\n" << h.t() << "\n";
  //Rcout << "r:\n" << r.t() << "\n";
  
  
  
  for (int j = 0; j < p; j++) {
    xwr = arma::accu(X.col(j) % r % h);
    //Rcout << "xwr " << j << ": " << xwr << "\n";
    xwx = arma::accu(h % arma::square(X.col(j)));
    //Rcout << "xwx " << j << ": " << xwx << "\n";
    u = xwr / n + (xwx / n) * a[j];
    v = xwx / n;
    //Rcout << "u " << j << ": " << u << "\n";
    //Rcout << "v " << j << ": " << v << "\n";
    
    l1 = lambda * m[j] * alpha;
    l2 = lambda * m[j] * (1 - alpha);
    //Rcout << "l1 " << j << ": " << l1 << "\n";
    //Rcout << "l2 " << j << ": " << l2 << "\n";
    // if (penalty == "MCP") b[j] = MCP(u, l1, l2, gamma, v);
    // if (penalty == "SCAD") b[j] = SCAD(u, l1, l2, gamma, v);
    b[j] = lasso(u, l1, l2, v);
    //Rcout << "b " << j << ": " << b << "\n";
    
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
                          double alpha, double lambda, arma::vec beta0, int it,
                          bool& flag_nan){

  // Order y by time
  //Rcout << "test a";
  arma::uvec tOrder = arma::sort_index(y.col(0));
  arma::vec yy = arma::conv_to<arma::vec>::from(y.col(0)).elem(tOrder);
  arma::vec Delta = arma::conv_to<arma::vec>::from(y.col(1)).elem(tOrder);
  arma::mat XX = X.rows(tOrder);
  // Standardize X
  arma::rowvec meanX = arma::mean(XX, 0);
  arma::rowvec sdX = arma::stddev(XX, 1, 0); // using unbiased estimator
  //Rcout << sdX << "\n";
  XX.each_row() -= meanX;
  //XX.each_row() /= sdX;
  arma::uvec temp = arma::find_nan(XX);
  //Rcout << "number of nan in XX: " << temp.n_elem << "\n";
  //Rcout << XX.rows(0,4) << "\n";
  arma::uvec ns = arma::find(sdX > .000001);
  //Rcout << "test b";
  arma::colvec sdXt = sdX.t();
  sdXt = sdXt.elem(ns);
  XX = XX.cols(ns);
  XX.each_row() /= sdXt.t();
  beta0=beta0.elem(ns);
  int p = XX.n_cols;
  //Rcout << "test c";
  arma::vec penalty_factor = arma::ones<arma::vec>(p);
  
  //Rcout << "XX"<< XX * beta0 << "\n";
  
  // perform coordinate descent
  arma::vec b = cdfit_cox_dh_one_lambda_it(XX, Delta, lambda,
                                           beta0, penalty_factor, alpha, it, flag_nan);
  //Rcout << "test d";
  // Unstandardize coefficients
  arma::vec beta = arma::zeros<arma::vec>(X.n_cols);
  arma::vec bb = b / sdXt;
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
                       double lambdaW, double lambdaH,
                       double tol, int maxit, bool verbose, bool init){
  arma::mat H = H0;
  arma::mat W = W0;
  arma::colvec beta = beta0;
  
  List l;
  
  double std_nmf = 1;
  double std_surv = 1;
  
//Rcout << "std_nmf: " << std_nmf << "\n";
//Rcout << "std_surv: " << std_surv << "\n";
  
  int N = H.n_cols;
  int P = X.n_rows;
  int k = H.n_rows;
  
  //lambda = lambda/k;
  //lambdaW = lambdaW/k;
  //lambdaH = lambdaH/k;
  
  double loss = 0.000001;
  double eps = 1;
  int it = -1;
  double loss_prev;
  arma::mat s = arma::join_horiz(y,delta);
  arma::mat W_prev;
  bool flag_nan=FALSE;

  arma::vec lossit = arma::zeros<arma::vec>(maxit);
  arma::vec slossit = arma::zeros<arma::vec>(maxit);
  arma::vec nlossit = arma::zeros<arma::vec>(maxit);

  while(eps > tol && it < maxit){
    it = it + 1;
   //Rcout << "loss: " << loss << "\n";
    loss_prev = loss;
    
    W_prev=W;
    
    update_W_cpp(X, M, y, delta, W, H, beta, alpha, std_nmf, std_surv, lambdaW);
    //Rcout << "W:\n" << W.rows(0,8) << "\n";
    arma::uvec temp = arma::find_nan(W);
    //Rcout << "number of nan:" << temp.n_elem << "\n";
    //Rcout << "test1";

    beta = update_beta_cpp(trans(M % X) * W, s, eta, lambda, beta, it, flag_nan);
    
    //Rcout << "beta: " << beta << "\n";
    
    if(flag_nan){
      W=W_prev;
      it=it-1;
      break;
    }
    //Rcout << "test2";
    //Rcout << "beta:\n" << beta << "\n";


    update_H_cpp(X, M, y, delta, W, H, alpha, std_nmf, lambdaH);
    //Rcout << "H:\n" << H << "\n";
    //Rcout << "test3";
    //standardize(W,H,beta);
    ////Rcout << "test4\n";
    
    l = calc_loss_cpp(X, M, y, delta, W, H, beta, alpha, lambda, eta, 
                      std_nmf, std_surv, lambdaW, lambdaH);

    loss = l["loss"];
    //Rcout << "loss: " << loss << "\n";
    
    double survloss = l["surv_loss"];
   //Rcout << "surv loss: " << survloss << "\n";
    double nmfloss = l["nmf_loss"];
   //Rcout << "nmf loss: " << nmfloss << "\n";
    double penloss = l["penalty"];
    
    double penaltyW = l["penalty_W"];
    double penaltyH = l["penalty_H"];
    double penaltybeta = l["penalty_beta"];
    
    //Rcout << "survloss: " << survloss*alpha << " nmfloss" << nmfloss*(1-alpha) << "\npenalty W: " << penaltyW << " penalty H: " << penaltyH << " penalty beta: " << penaltybeta << "\n";
    
    if(it==0){
      std_nmf = nmfloss;
      std_surv = std::abs(survloss);
    }
    
    // Rcout << "loss\n" << loss << "\n";
    // Rcout << "surv loss\n" << survloss << "\n";
    // Rcout << "nmf loss\n" << nmfloss << "\n";
    // Rcout << "penalty\n" << penloss << "\n";
    // 
    // if(it>1600){
    //Rcout << "H\n" << H.cols(0,8) << "\n";
    // }
    
    // 
    // Rcout << "beta\n" << beta << "\n";
    // Rcout << "lp\n" << lptemp.rows(0,4) << "\n";

    eps = std::abs(loss - loss_prev)/loss_prev;
    if(it>0){
      lossit[it-1] = loss;
      nlossit[it-1] = nmfloss;
      slossit[it-1] = survloss;
    }

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
    Named("lossit") = lossit,
    Named("slossit") = slossit,
    Named("nlossit") = nlossit,
    Named("convergence") = it<maxit,
    Named("NaN flag") = flag_nan);
  return L;
}
