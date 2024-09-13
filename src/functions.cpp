#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <cmath>
#include <iostream>
#include <stdexcept>  // std::invalid_argument
#include <vector>
#include "LBFGSpp/Param.h"
#include "LBFGSpp/BFGSMat.h"
#include "LBFGSpp/Cauchy.h"
#include "LBFGSpp/SubspaceMin.h"
#include "LBFGSpp/LineSearchMoreThuente.h"

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using Eigen::VectorXd;

namespace LBFGSpp {

///
/// L-BFGS-B solver for box-constrained numerical optimization
///
template <typename Scalar,
          template <class> class LineSearch = LineSearchMoreThuente>
class LBFGSBSolver
{
private:
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using MapVec = Eigen::Map<Vector>;
  using IndexSet = std::vector<int>;
  
  const LBFGSBParam<Scalar>& m_param;  // Parameters to control the LBFGS algorithm
  BFGSMat<Scalar, true> m_bfgs;        // Approximation to the Hessian matrix
  Vector m_fx;                         // History of the objective function values
  Vector m_xp;                         // Old x
  Vector m_grad;                       // New gradient
  Scalar m_projgnorm;                  // Projected gradient norm
  Vector m_gradp;                      // Old gradient
  Vector m_drt;                        // Moving direction
  
  // Reset internal variables
  // n: dimension of the vector to be optimized
  inline void reset(int n)
  {
    const int m = m_param.m;
    m_bfgs.reset(n, m);
    m_xp.resize(n);
    m_grad.resize(n);
    m_gradp.resize(n);
    m_drt.resize(n);
    if (m_param.past > 0)
      m_fx.resize(m_param.past);
  }
  
  // Project the vector x to the bound constraint set
  static void force_bounds(Vector& x, const Vector& lb, const Vector& ub)
  {
    x.noalias() = x.cwiseMax(lb).cwiseMin(ub);
  }
  
  // Norm of the projected gradient
  // ||P(x-g, l, u) - x||_inf
  static Scalar proj_grad_norm(const Vector& x, const Vector& g, const Vector& lb, const Vector& ub)
  {
    return ((x - g).cwiseMax(lb).cwiseMin(ub) - x).cwiseAbs().maxCoeff();
  }
  
  // The maximum step size alpha such that x0 + alpha * d stays within the bounds
  static Scalar max_step_size(const Vector& x0, const Vector& drt, const Vector& lb, const Vector& ub)
  {
    const int n = x0.size();
    Scalar step = std::numeric_limits<Scalar>::infinity();
    
    for (int i = 0; i < n; i++)
    {
      if (drt[i] > Scalar(0))
      {
        step = std::min(step, (ub[i] - x0[i]) / drt[i]);
      }
      else if (drt[i] < Scalar(0))
      {
        step = std::min(step, (lb[i] - x0[i]) / drt[i]);
      }
    }
    
    return step;
  }
  
public:
  ///
  /// Constructor for the L-BFGS-B solver.
  ///
  /// \param param An object of \ref LBFGSParam to store parameters for the
  ///        algorithm
  ///
  LBFGSBSolver(const LBFGSBParam<Scalar>& param) :
  m_param(param)
  {
    m_param.check_param();
  }
  
  ///
  /// Minimizing a multivariate function subject to box constraints, using the L-BFGS-B algorithm.
  /// Exceptions will be thrown if error occurs.
  ///
  /// \param f  A function object such that `f(x, grad)` returns the
  ///           objective function value at `x`, and overwrites `grad` with
  ///           the gradient.
  /// \param x  In: An initial guess of the optimal point. Out: The best point
  ///           found.
  /// \param fx Out: The objective function value at `x`.
  /// \param lb Lower bounds for `x`.
  /// \param ub Upper bounds for `x`.
  ///
  /// \return Number of iterations used.
  ///
  template <typename Foo>
  inline int minimize(Foo& f, Vector& x, Scalar& fx, const Vector& lb, const Vector& ub)
  {
    using std::abs;
    
    // Dimension of the vector
    const int n = x.size();
    if (lb.size() != n || ub.size() != n)
      throw std::invalid_argument("'lb' and 'ub' must have the same size as 'x'");
    
    // Check whether the initial vector is within the bounds
    // If not, project to the feasible set
    force_bounds(x, lb, ub);
    
    // Initialization
    reset(n);
    
    // The length of lag for objective function value to test convergence
    const int fpast = m_param.past;
    
    // Evaluate function and compute gradient
    fx = f(x, m_grad);
    m_projgnorm = proj_grad_norm(x, m_grad, lb, ub);
    if (fpast > 0)
      m_fx[0] = fx;
    
    // std::cout << "x0 = " << x.transpose() << std::endl;
    // std::cout << "f(x0) = " << fx << ", ||proj_grad|| = " << m_projgnorm << std::endl << std::endl;
    
    // Early exit if the initial x is already a minimizer
    if (m_projgnorm <= m_param.epsilon || m_projgnorm <= m_param.epsilon_rel * x.norm())
    {
      return 1;
    }
    
    // Compute generalized Cauchy point
    Vector xcp(n), vecc;
    IndexSet newact_set, fv_set;
    Cauchy<Scalar>::get_cauchy_point(m_bfgs, x, m_grad, lb, ub, xcp, vecc, newact_set, fv_set);
    
    /* Vector gcp(n);
     Scalar fcp = f(xcp, gcp);
     Scalar projgcpnorm = proj_grad_norm(xcp, gcp, lb, ub);
     std::cout << "xcp = " << xcp.transpose() << std::endl;
     std::cout << "f(xcp) = " << fcp << ", ||proj_grad|| = " << projgcpnorm << std::endl << std::endl; */
    
    // Initial direction
    m_drt.noalias() = xcp - x;
    m_drt.normalize();
    // Tolerance for s'y >= eps * (y'y)
    constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
    // s and y vectors
    Vector vecs(n), vecy(n);
    // Number of iterations used
    int k = 1;
    for (;;)
    {
      // Save the curent x and gradient
      m_xp.noalias() = x;
      m_gradp.noalias() = m_grad;
      Scalar dg = m_grad.dot(m_drt);
      
      // Maximum step size to make x feasible
      Scalar step_max = max_step_size(x, m_drt, lb, ub);
      
      // In some cases, the direction returned by the subspace minimization procedure
      // in the previous iteration is pathological, leading to issues such as
      // step_max~=0 and dg>=0. If this happens, we use xcp-x as the search direction,
      // and reset the BFGS matrix. This is because xsm (the subspace minimizer)
      // heavily depends on the BFGS matrix. If xsm is corrupted, then we may suspect
      // there is something wrong in the BFGS matrix, and it is safer to reset the matrix.
      // In contrast, xcp is obtained from a line search, which tends to be more robust
      if (dg >= Scalar(0) || step_max <= m_param.min_step)
      {
        // Reset search direction
        m_drt.noalias() = xcp - x;
        // Reset BFGS matrix
        m_bfgs.reset(n, m_param.m);
        // Recompute dg and step_max
        dg = m_grad.dot(m_drt);
        step_max = max_step_size(x, m_drt, lb, ub);
      }
      
      // Line search to update x, fx and gradient
      step_max = std::min(m_param.max_step, step_max);
      Scalar step = Scalar(1);
      step = std::min(step, step_max);
      LineSearch<Scalar>::LineSearch(f, m_param, m_xp, m_drt, step_max, step, fx, m_grad, dg, x);
      
      // New projected gradient norm
      m_projgnorm = proj_grad_norm(x, m_grad, lb, ub);
      
      /* std::cout << "** Iteration " << k << std::endl;
       std::cout << "   x = " << x.transpose() << std::endl;
       std::cout << "   f(x) = " << fx << ", ||proj_grad|| = " << m_projgnorm << std::endl << std::endl; */
      
      // Convergence test -- gradient
      if (m_projgnorm <= m_param.epsilon || m_projgnorm <= m_param.epsilon_rel * x.norm())
      {
        return k;
      }
      // Convergence test -- objective function value
      if (fpast > 0)
      {
        const Scalar fxd = m_fx[k % fpast];
        if (k >= fpast && abs(fxd - fx) <= m_param.delta * std::max(std::max(abs(fx), abs(fxd)), Scalar(1)))
          return k;
        
        m_fx[k % fpast] = fx;
      }
      // Maximum number of iterations
      if (m_param.max_iterations != 0 && k >= m_param.max_iterations)
      {
        return k;
      }
      
      // Update s and y
      // s_{k+1} = x_{k+1} - x_k
      // y_{k+1} = g_{k+1} - g_k
      vecs.noalias() = x - m_xp;
      vecy.noalias() = m_grad - m_gradp;
      if (vecs.dot(vecy) > eps * vecy.squaredNorm())
        m_bfgs.add_correction(vecs, vecy);
      
      force_bounds(x, lb, ub);
      Cauchy<Scalar>::get_cauchy_point(m_bfgs, x, m_grad, lb, ub, xcp, vecc, newact_set, fv_set);
      
      /*Vector gcp(n);
       Scalar fcp = f(xcp, gcp);
       Scalar projgcpnorm = proj_grad_norm(xcp, gcp, lb, ub);
       std::cout << "xcp = " << xcp.transpose() << std::endl;
       std::cout << "f(xcp) = " << fcp << ", ||proj_grad|| = " << projgcpnorm << std::endl << std::endl;*/
      
      SubspaceMin<Scalar>::subspace_minimize(m_bfgs, x, xcp, m_grad, lb, ub,
                                             vecc, newact_set, fv_set, m_param.max_submin, m_drt);
      
      /*Vector gsm(n);
       Scalar fsm = f(x + m_drt, gsm);
       Scalar projgsmnorm = proj_grad_norm(x + m_drt, gsm, lb, ub);
       std::cout << "xsm = " << (x + m_drt).transpose() << std::endl;
       std::cout << "f(xsm) = " << fsm << ", ||proj_grad|| = " << projgsmnorm << std::endl << std::endl;*/
      
      k++;
    }
    
    return k;
  }
  
  ///
  /// Returning the gradient vector on the last iterate.
  /// Typically used to debug and test convergence.
  /// Should only be called after the `minimize()` function.
  ///
  /// \return A const reference to the gradient vector.
  ///
  const Vector& final_grad() const { return m_grad; }
  
  ///
  /// Returning the infinity norm of the final projected gradient.
  /// The projected gradient is defined as \f$P(x-g,l,u)-x\f$, where \f$P(v,l,u)\f$ stands for
  /// the projection of a vector \f$v\f$ onto the box specified by the lower bound vector \f$l\f$ and
  /// upper bound vector \f$u\f$.
  ///
  Scalar final_grad_norm() const { return m_projgnorm; }
};

}  // namespace LBFGSpp



// [[Rcpp::export]]
void update_H_cpp(const arma::mat& X, const arma::mat& M, const arma::mat& W,
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
    arma::mat I = arma::conv_to<arma::mat>::from(y_matrix >= y_matrix.t());
    
    // intermediate matrix
    arma::mat temp = I.t() % arma::repmat(lp.t(),N,1) / arma::repmat(I.t() * lp, 1, N);
    
    // derivative of log likelihood
    arma::rowvec delta_t = delta.t();
    arma::mat l = arma::kron(delta.t() - (delta.t() * temp),beta);
    
    // H update
    H = (H / (Wt * (M % (W * H)))) % 
      arma::clamp((Wt * (M % X)) + (alpha * arma::accu(M) / N) * 
      l,0,arma::datum::inf);
  }
  
  return;
}

// [[Rcpp::export]]
void update_W_cpp(const arma::mat& X, const arma::mat& M, const arma::mat& H,
                       arma::mat& W, const arma::colvec& beta, 
                       const arma::colvec& y, const arma::colvec& delta, 
                       double alpha, bool WtX, int norm_type) {
  if(!WtX){
    W = W % ((M % X) * H.t()) / ((M % (W * H)) * H.t());
  }else{
    int N = H.n_cols;
    int s = arma::accu(M);
    
    // linear predictor
    arma::vec lp = exp(trans(M % X) * W * beta);

    // Indicator matrix
    arma::mat y_matrix = arma::repmat(y, 1, N);
    arma::mat I = arma::conv_to<arma::mat>::from(y_matrix >= y_matrix.t());

    // derivative of log likelihood
    arma::mat P = diagmat(lp);
    arma::mat l = arma::kron((M % X) * (arma::eye(N,N) - P*I*inv(diagmat(I.t() * lp))) * delta,beta.t());

    arma::mat Ht = H.t();
    // arma::mat temp1 = (M % X) * Ht;
    // arma::mat temp2 = (alpha * s / N) * l;
    // arma::mat temp3 = temp1 + temp2;
    // Rcout << temp1.rows(0,10) << "\n\n";
    // Rcout << temp2.rows(0,10) << "\n\n";
    // Rcout << temp3.rows(0,10) << "\n\n";
    W = W % (arma::clamp((M % X) * Ht + (alpha * s / N) * l, 0, arma::datum::inf) / ((M % (W*H)) * Ht));
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
double calc_surv_loss(const arma::mat& X, const arma::mat& M, const arma::mat& W,
                      const arma::mat& H, const arma::vec& beta, const arma::vec& y,
                      const arma::vec& delta, bool WtX){
  int N = H.n_cols;
  arma::colvec lp;
  
  if(!WtX){
    lp = H.t() * beta;
  }else{
    lp = trans(M % X) * W * beta;
  }
  
  arma::mat y_matrix = arma::repmat(y, 1, N);
  arma::mat ind = arma::conv_to<arma::mat>::from(y_matrix >= y_matrix.t());
  
  return 2 * arma::accu(delta % (lp - arma::log(ind.t() * arma::exp(lp)))) / N;
  
}

// [[Rcpp::export]]
List calc_loss_cpp(const arma::mat& X, const arma::mat& M, const arma::mat& W, const arma::mat& H,
                   const arma::vec& beta, double alpha, const arma::vec& y, 
                   const arma::vec& delta, double lambda, double eta, bool WtX) {
  
  
  double nmf_loss = arma::accu(arma::square(M % (X - W * H))) / arma::accu(M);
  double surv_loss = calc_surv_loss(X, M, W, H, beta, y, delta, WtX);
  double penalty = lambda * ((1 - eta) * arma::accu(arma::square(beta)) / 2 + eta * arma::accu(arma::abs(beta)));
  double loss = nmf_loss - alpha * (surv_loss - penalty);
  
  return List::create(
    Named("loss") = loss,
    Named("nmf_loss") = nmf_loss,
    Named("surv_loss") = surv_loss,
    Named("penalty") = penalty
  );
}

class Hupdate
{
private:
  const arma::mat& M;
  const arma::mat& X;
  const arma::vec& y;
  const arma::vec& delta;
  arma::mat& W;
  arma::vec& beta;
  double alpha;
  double lambda;
  double eta;
  bool WtX;
public:
  
  Hupdate(const arma::mat& M_, const arma::mat& X_, const arma::vec& y_,
          const arma::vec& delta_, arma::mat& W_,
          arma::vec& beta_, double alpha_, double lambda_,
          double eta_, bool WtX_) : M(M_), X(X_), y(y_), delta(delta_), W(W_), beta(beta_),
          alpha(alpha_), lambda(lambda_), eta(eta_), WtX(WtX_) {}
  double operator()(const VectorXd& x, VectorXd& grad)
  {
    double fx = 0.0;
    
    
    int s = arma::accu(M);
    int k = W.n_cols;
    int n = X.n_cols;
    
    //create H from x
    //convert eigen vector x to std vector
    std::vector<double> xstd(x.data(), x.data() + x.size());
    //convert std vector to arma vector
    arma::vec xarma = arma::conv_to< arma::colvec >::from(xstd);
    //create H by stacking arma vector into columns
    arma::mat H = arma::reshape(xarma,k,n);
    int N = H.n_cols;
    
    // COMPUTE GRADIENT
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
    // compute gradient in matrix form
    arma::mat nmf = W.t() * (M % (W*H - X)) * (2.0/s);
    arma::mat like = alpha * 2 * l / N;
    arma::mat gradient = nmf - like;
    // convert to armadillo vector
    arma::vec v = arma::vectorise(gradient);
    // convert to standard vector
    std::vector<double> v2 = arma::conv_to < std::vector<double> >::from(v);
    // convert to eigen vectorXd
    grad = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(v2.data(), v2.size());
    
    // Rcout << "nmf:\n" << nmf.submat(0,0,5-1,0) << "\n";
    // Rcout << "like:\n" << like.submat(0,0,5-1,0) << "\n";
    // Rcout << "grad:\n" << gradient.submat(0,0,5-1,0) << "\n";
    // Rcout << "s:\n" << s << "\n";
    
    //COMPUTE FUNCTION VALUE
    List loss = calc_loss_cpp(X,M,W,H,beta,alpha,y,delta,lambda,eta,WtX);
    fx = loss["loss"];
    return fx;
  }
  
  void set_value(arma::mat& W_, arma::mat& beta_){
    beta=beta_;
    W=W_;
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
                          double alpha, double lambda, arma::vec beta0){


  // Order y by time
  arma::uvec tOrder = arma::sort_index(y.col(0));
  arma::vec yy = arma::conv_to<arma::vec>::from(y.col(0)).elem(tOrder);
  arma::vec Delta = arma::conv_to<arma::vec>::from(y.col(1)).elem(tOrder);
  arma::mat XX = X.rows(tOrder);
  // Standardize X
  arma::rowvec meanX = arma::mean(XX, 0);
  arma::rowvec sdX = arma::stddev(XX, 1, 0); // using unbiased estimator
  XX.each_row() -= meanX;
  XX.each_row() /= sdX;
  arma::uvec ns = arma::find(sdX > .000001);
  XX = XX.cols(ns);
  int p = XX.n_cols;

  arma::vec penalty_factor = arma::ones<arma::vec>(p);
  penalty_factor = penalty_factor.elem(ns);
  
  // perform coordinate descent
  arma::vec b = cdfit_cox_dh_one_lambda_it(XX, Delta, penalty, lambda,
                                           beta0, penalty_factor, alpha);

  // Unstandardize coefficients
  arma::vec beta = arma::zeros<arma::vec>(X.n_cols);
  arma::vec bb = b / sdX.t();
  beta.elem(ns) = bb;

  return beta;
}

//' @export
// [[Rcpp::export]]
void standardize(arma::mat& W, arma::mat& H, arma::colvec& beta, int norm_type,
                 bool WtX){
  
  arma::rowvec col_max = max(W, 0);
  W.each_row() /= col_max;
  H.each_col() %= col_max.t();
  
  if(WtX){
    beta %= col_max.t();
  }else{
    beta /= col_max.t();
  }
  
  
    // arma::rowvec col_sums = sum(W, 0);
    // W.each_row() /= col_sums;
    // H.each_col() %= col_sums.t();
    // beta /= col_sums.t();
    
  // arma::colvec row_sums = sum(W,1);
  // W.each_col() /= row_sums;
  // H.each_row() 
  
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
                            String penalty, bool init){
  arma::mat H = H0;
  arma::mat W = W0;
  
  arma::colvec beta = beta0;
  
  int N = H.n_cols;
  int k = H.n_rows;
  
  double loss = 0.000001;
  double eps = 1;
  int it = 0;
  double loss_prev;
  List l;
  arma::mat s = arma::join_horiz(y,delta);
  
  LBFGSpp::LBFGSBParam<double> param;
  param.epsilon = 1e-6;
  param.max_iterations = 100;
  double fx;
  // Create solver and function object
  LBFGSpp::LBFGSBSolver<double> solver(param);
  // Declare function object Hupdate
  Hupdate fun(M,X,y,delta,W,beta,alpha,lambda,eta,WtX);
  
  // bounds for constrained optimization
  VectorXd lb = VectorXd::Constant(N*k, 0.0);
  VectorXd ub = VectorXd::Constant(N*k, std::numeric_limits<double>::infinity());

  // declare intermediate vectors
  arma::vec xarma;
  std::vector<double> xstd;
  VectorXd x;
  std::vector<double> xstd2;
  arma::vec xarma2;
  
  while(eps > tol && it <= maxit){
    loss_prev = loss;
    update_W_cpp(X,M,H,W,beta,y,delta,alpha,WtX,norm_type);
    if(WtX){
      beta = update_beta_cpp(trans(M % X) * W, s,penalty,eta,lambda,beta);
    }else{
      beta = update_beta_cpp(H.t(),s,penalty,eta,lambda,beta);
    }

    fun.set_value(W,beta);
    
    //update_H_cpp(X,M,W,beta,H,y,delta,alpha,WtX);
    // convert H to arma::vec
    xarma = arma::vectorise(H);
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
    H = arma::reshape(xarma2,k,N);
    

    // standardize
    standardize(W,H,beta,norm_type,WtX);
    
    

    l = calc_loss_cpp(X, M, W, H, beta, alpha, y, delta, lambda, eta, WtX);
    loss = l["loss"];

    eps = std::abs(loss - loss_prev)/loss_prev;

    it = it + 1;
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
    Named("iter") = it);
  return L;
}
