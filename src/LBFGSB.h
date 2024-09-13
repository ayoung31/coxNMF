// Copyright (C) 2020-2023 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSPP_LBFGSB_H
#define LBFGSPP_LBFGSB_H

#include <stdexcept>  // std::invalid_argument
#include <vector>
#include <Eigen/Core>



namespace LBFGSpp {

///
/// \defgroup Enumerations
///
/// Enumeration types for line search.
///

///
/// \ingroup Enumerations
///
/// The enumeration of line search termination conditions.
///
enum LINE_SEARCH_TERMINATION_CONDITION
{
  ///
  /// Backtracking method with the Armijo condition.
  /// The backtracking method finds the step length such that it satisfies
  /// the sufficient decrease (Armijo) condition,
  /// \f$f(x + a \cdot d) \le f(x) + \beta' \cdot a \cdot g(x)^T d\f$,
  /// where \f$x\f$ is the current point, \f$d\f$ is the current search direction,
  /// \f$a\f$ is the step length, and \f$\beta'\f$ is the value specified by
  /// \ref LBFGSParam::ftol. \f$f\f$ and \f$g\f$ are the function
  /// and gradient values respectively.
  ///
  LBFGS_LINESEARCH_BACKTRACKING_ARMIJO = 1,
  
  ///
  /// The backtracking method with the defualt (regular Wolfe) condition.
  /// An alias of `LBFGS_LINESEARCH_BACKTRACKING_WOLFE`.
  ///
  LBFGS_LINESEARCH_BACKTRACKING = 2,
  
  ///
  /// Backtracking method with regular Wolfe condition.
  /// The backtracking method finds the step length such that it satisfies
  /// both the Armijo condition (`LBFGS_LINESEARCH_BACKTRACKING_ARMIJO`)
  /// and the curvature condition,
  /// \f$g(x + a \cdot d)^T d \ge \beta \cdot g(x)^T d\f$, where \f$\beta\f$
  /// is the value specified by \ref LBFGSParam::wolfe.
  ///
  LBFGS_LINESEARCH_BACKTRACKING_WOLFE = 2,
  
  ///
  /// Backtracking method with strong Wolfe condition.
  /// The backtracking method finds the step length such that it satisfies
  /// both the Armijo condition (`LBFGS_LINESEARCH_BACKTRACKING_ARMIJO`)
  /// and the following condition,
  /// \f$\vert g(x + a \cdot d)^T d\vert \le \beta \cdot \vert g(x)^T d\vert\f$,
  /// where \f$\beta\f$ is the value specified by \ref LBFGSParam::wolfe.
  ///
  LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3
};

///
/// Parameters to control the L-BFGS algorithm.
///
template <typename Scalar = double>
class LBFGSParam
{
public:
  ///
  /// The number of corrections to approximate the inverse Hessian matrix.
  /// The L-BFGS routine stores the computation results of previous \ref m
  /// iterations to approximate the inverse Hessian matrix of the current
  /// iteration. This parameter controls the size of the limited memories
  /// (corrections). The default value is \c 6. Values less than \c 3 are
  /// not recommended. Large values will result in excessive computing time.
  ///
  int m;
  ///
  /// Absolute tolerance for convergence test.
  /// This parameter determines the absolute accuracy \f$\epsilon_{abs}\f$
  /// with which the solution is to be found. A minimization terminates when
  /// \f$||g|| < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
  /// where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
  /// \c 1e-5.
  ///
  Scalar epsilon;
  ///
  /// Relative tolerance for convergence test.
  /// This parameter determines the relative accuracy \f$\epsilon_{rel}\f$
  /// with which the solution is to be found. A minimization terminates when
  /// \f$||g|| < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
  /// where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
  /// \c 1e-5.
  ///
  Scalar epsilon_rel;
  ///
  /// Distance for delta-based convergence test.
  /// This parameter determines the distance \f$d\f$ to compute the
  /// rate of decrease of the objective function,
  /// \f$f_{k-d}(x)-f_k(x)\f$, where \f$k\f$ is the current iteration
  /// step. If the value of this parameter is zero, the delta-based convergence
  /// test will not be performed. The default value is \c 0.
  ///
  int past;
  ///
  /// Delta for convergence test.
  /// The algorithm stops when the following condition is met,
  /// \f$|f_{k-d}(x)-f_k(x)|<\delta\cdot\max(1, |f_k(x)|, |f_{k-d}(x)|)\f$, where \f$f_k(x)\f$ is
  /// the current function value, and \f$f_{k-d}(x)\f$ is the function value
  /// \f$d\f$ iterations ago (specified by the \ref past parameter).
  /// The default value is \c 0.
  ///
  Scalar delta;
  ///
  /// The maximum number of iterations.
  /// The optimization process is terminated when the iteration count
  /// exceeds this parameter. Setting this parameter to zero continues an
  /// optimization process until a convergence or error. The default value
  /// is \c 0.
  ///
  int max_iterations;
  ///
  /// The line search termination condition.
  /// This parameter specifies the line search termination condition that will be used
  /// by the LBFGS routine. The default value is `LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE`.
  ///
  int linesearch;
  ///
  /// The maximum number of trials for the line search.
  /// This parameter controls the number of function and gradients evaluations
  /// per iteration for the line search routine. The default value is \c 20.
  ///
  int max_linesearch;
  ///
  /// The minimum step length allowed in the line search.
  /// The default value is \c 1e-20. Usually this value does not need to be
  /// modified.
  ///
  Scalar min_step;
  ///
  /// The maximum step length allowed in the line search.
  /// The default value is \c 1e+20. Usually this value does not need to be
  /// modified.
  ///
  Scalar max_step;
  ///
  /// A parameter to control the accuracy of the line search routine.
  /// The default value is \c 1e-4. This parameter should be greater
  /// than zero and smaller than \c 0.5.
  ///
  Scalar ftol;
  ///
  /// The coefficient for the Wolfe condition.
  /// This parameter is valid only when the line-search
  /// algorithm is used with the Wolfe condition.
  /// The default value is \c 0.9. This parameter should be greater
  /// the \ref ftol parameter and smaller than \c 1.0.
  ///
  Scalar wolfe;
  
public:
  ///
  /// Constructor for L-BFGS parameters.
  /// Default values for parameters will be set when the object is created.
  ///
  LBFGSParam()
  {
    // clang-format off
    m              = 6;
    epsilon        = Scalar(1e-5);
    epsilon_rel    = Scalar(1e-5);
    past           = 0;
    delta          = Scalar(0);
    max_iterations = 0;
    linesearch     = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
    max_linesearch = 20;
    min_step       = Scalar(1e-20);
    max_step       = Scalar(1e+20);
    ftol           = Scalar(1e-4);
    wolfe          = Scalar(0.9);
    // clang-format on
  }
  
  ///
  /// Checking the validity of L-BFGS parameters.
  /// An `std::invalid_argument` exception will be thrown if some parameter
  /// is invalid.
  ///
  inline void check_param() const
  {
    if (m <= 0)
      throw std::invalid_argument("'m' must be positive");
    if (epsilon < 0)
      throw std::invalid_argument("'epsilon' must be non-negative");
    if (epsilon_rel < 0)
      throw std::invalid_argument("'epsilon_rel' must be non-negative");
    if (past < 0)
      throw std::invalid_argument("'past' must be non-negative");
    if (delta < 0)
      throw std::invalid_argument("'delta' must be non-negative");
    if (max_iterations < 0)
      throw std::invalid_argument("'max_iterations' must be non-negative");
    if (linesearch < LBFGS_LINESEARCH_BACKTRACKING_ARMIJO ||
        linesearch > LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
      throw std::invalid_argument("unsupported line search termination condition");
    if (max_linesearch <= 0)
      throw std::invalid_argument("'max_linesearch' must be positive");
    if (min_step < 0)
      throw std::invalid_argument("'min_step' must be positive");
    if (max_step < min_step)
      throw std::invalid_argument("'max_step' must be greater than 'min_step'");
    if (ftol <= 0 || ftol >= 0.5)
      throw std::invalid_argument("'ftol' must satisfy 0 < ftol < 0.5");
    if (wolfe <= ftol || wolfe >= 1)
      throw std::invalid_argument("'wolfe' must satisfy ftol < wolfe < 1");
  }
};

///
/// Parameters to control the L-BFGS-B algorithm.
///
template <typename Scalar = double>
class LBFGSBParam
{
public:
  ///
  /// The number of corrections to approximate the inverse Hessian matrix.
  /// The L-BFGS-B routine stores the computation results of previous \ref m
  /// iterations to approximate the inverse Hessian matrix of the current
  /// iteration. This parameter controls the size of the limited memories
  /// (corrections). The default value is \c 6. Values less than \c 3 are
  /// not recommended. Large values will result in excessive computing time.
  ///
  int m;
  ///
  /// Absolute tolerance for convergence test.
  /// This parameter determines the absolute accuracy \f$\epsilon_{abs}\f$
  /// with which the solution is to be found. A minimization terminates when
  /// \f$||Pg||_{\infty} < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
  /// where \f$||x||\f$ denotes the Euclidean (L2) norm of \f$x\f$, and
  /// \f$Pg=P(x-g,l,u)-x\f$ is the projected gradient. The default value is
  /// \c 1e-5.
  ///
  Scalar epsilon;
  ///
  /// Relative tolerance for convergence test.
  /// This parameter determines the relative accuracy \f$\epsilon_{rel}\f$
  /// with which the solution is to be found. A minimization terminates when
  /// \f$||Pg||_{\infty} < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
  /// where \f$||x||\f$ denotes the Euclidean (L2) norm of \f$x\f$, and
  /// \f$Pg=P(x-g,l,u)-x\f$ is the projected gradient. The default value is
  /// \c 1e-5.
  ///
  Scalar epsilon_rel;
  ///
  /// Distance for delta-based convergence test.
  /// This parameter determines the distance \f$d\f$ to compute the
  /// rate of decrease of the objective function,
  /// \f$f_{k-d}(x)-f_k(x)\f$, where \f$k\f$ is the current iteration
  /// step. If the value of this parameter is zero, the delta-based convergence
  /// test will not be performed. The default value is \c 1.
  ///
  int past;
  ///
  /// Delta for convergence test.
  /// The algorithm stops when the following condition is met,
  /// \f$|f_{k-d}(x)-f_k(x)|<\delta\cdot\max(1, |f_k(x)|, |f_{k-d}(x)|)\f$, where \f$f_k(x)\f$ is
  /// the current function value, and \f$f_{k-d}(x)\f$ is the function value
  /// \f$d\f$ iterations ago (specified by the \ref past parameter).
  /// The default value is \c 1e-10.
  ///
  Scalar delta;
  ///
  /// The maximum number of iterations.
  /// The optimization process is terminated when the iteration count
  /// exceeds this parameter. Setting this parameter to zero continues an
  /// optimization process until a convergence or error. The default value
  /// is \c 0.
  ///
  int max_iterations;
  ///
  /// The maximum number of iterations in the subspace minimization.
  /// This parameter controls the number of iterations in the subspace
  /// minimization routine. The default value is \c 10.
  ///
  int max_submin;
  ///
  /// The maximum number of trials for the line search.
  /// This parameter controls the number of function and gradients evaluations
  /// per iteration for the line search routine. The default value is \c 20.
  ///
  int max_linesearch;
  ///
  /// The minimum step length allowed in the line search.
  /// The default value is \c 1e-20. Usually this value does not need to be
  /// modified.
  ///
  Scalar min_step;
  ///
  /// The maximum step length allowed in the line search.
  /// The default value is \c 1e+20. Usually this value does not need to be
  /// modified.
  ///
  Scalar max_step;
  ///
  /// A parameter to control the accuracy of the line search routine.
  /// The default value is \c 1e-4. This parameter should be greater
  /// than zero and smaller than \c 0.5.
  ///
  Scalar ftol;
  ///
  /// The coefficient for the Wolfe condition.
  /// This parameter is valid only when the line-search
  /// algorithm is used with the Wolfe condition.
  /// The default value is \c 0.9. This parameter should be greater
  /// the \ref ftol parameter and smaller than \c 1.0.
  ///
  Scalar wolfe;
  
public:
  ///
  /// Constructor for L-BFGS-B parameters.
  /// Default values for parameters will be set when the object is created.
  ///
  LBFGSBParam()
  {
    // clang-format off
    m              = 6;
    epsilon        = Scalar(1e-5);
    epsilon_rel    = Scalar(1e-5);
    past           = 1;
    delta          = Scalar(1e-10);
    max_iterations = 0;
    max_submin     = 10;
    max_linesearch = 20;
    min_step       = Scalar(1e-20);
    max_step       = Scalar(1e+20);
    ftol           = Scalar(1e-4);
    wolfe          = Scalar(0.9);
    // clang-format on
  }
  
  ///
  /// Checking the validity of L-BFGS-B parameters.
  /// An `std::invalid_argument` exception will be thrown if some parameter
  /// is invalid.
  ///
  inline void check_param() const
  {
    if (m <= 0)
      throw std::invalid_argument("'m' must be positive");
    if (epsilon < 0)
      throw std::invalid_argument("'epsilon' must be non-negative");
    if (epsilon_rel < 0)
      throw std::invalid_argument("'epsilon_rel' must be non-negative");
    if (past < 0)
      throw std::invalid_argument("'past' must be non-negative");
    if (delta < 0)
      throw std::invalid_argument("'delta' must be non-negative");
    if (max_iterations < 0)
      throw std::invalid_argument("'max_iterations' must be non-negative");
    if (max_submin < 0)
      throw std::invalid_argument("'max_submin' must be non-negative");
    if (max_linesearch <= 0)
      throw std::invalid_argument("'max_linesearch' must be positive");
    if (min_step < 0)
      throw std::invalid_argument("'min_step' must be positive");
    if (max_step < min_step)
      throw std::invalid_argument("'max_step' must be greater than 'min_step'");
    if (ftol <= 0 || ftol >= 0.5)
      throw std::invalid_argument("'ftol' must satisfy 0 < ftol < 0.5");
    if (wolfe <= ftol || wolfe >= 1)
      throw std::invalid_argument("'wolfe' must satisfy ftol < wolfe < 1");
  }
};

}  // namespace LBFGSpp



namespace LBFGSpp {

enum COMPUTATION_INFO
{
  SUCCESSFUL = 0,
  NOT_COMPUTED,
  NUMERICAL_ISSUE
};

// Bunch-Kaufman LDLT decomposition
// References:
// 1. Bunch, J. R., & Kaufman, L. (1977). Some stable methods for calculating inertia and solving symmetric linear systems.
//    Mathematics of computation, 31(137), 163-179.
// 2. Golub, G. H., & Van Loan, C. F. (2012). Matrix computations (Vol. 3). JHU press. Section 4.4.
// 3. Bunch-Parlett diagonal pivoting <http://oz.nthu.edu.tw/~d947207/Chap13_GE3.ppt>
// 4. Ashcraft, C., Grimes, R. G., & Lewis, J. G. (1998). Accurate symmetric indefinite linear equation solvers.
//    SIAM Journal on Matrix Analysis and Applications, 20(2), 513-561.
template <typename Scalar = double>
class BKLDLT
{
private:
  using Index = Eigen::Index;
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MapVec = Eigen::Map<Vector>;
  using MapConstVec = Eigen::Map<const Vector>;
  
  using IntVector = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
  using GenericVector = Eigen::Ref<Vector>;
  using GenericMatrix = Eigen::Ref<Matrix>;
  using ConstGenericMatrix = const Eigen::Ref<const Matrix>;
  using ConstGenericVector = const Eigen::Ref<const Vector>;
  
  Index m_n;
  Vector m_data;                                  // storage for a lower-triangular matrix
  std::vector<Scalar*> m_colptr;                  // pointers to columns
  IntVector m_perm;                               // [-2, -1, 3, 1, 4, 5]: 0 <-> 2, 1 <-> 1, 2 <-> 3, 3 <-> 1, 4 <-> 4, 5 <-> 5
  std::vector<std::pair<Index, Index> > m_permc;  // compressed version of m_perm: [(0, 2), (2, 3), (3, 1)]
  
  bool m_computed;
  int m_info;
  
  // Access to elements
  // Pointer to the k-th column
  Scalar* col_pointer(Index k) { return m_colptr[k]; }
  // A[i, j] -> m_colptr[j][i - j], i >= j
  Scalar& coeff(Index i, Index j) { return m_colptr[j][i - j]; }
  const Scalar& coeff(Index i, Index j) const { return m_colptr[j][i - j]; }
  // A[i, i] -> m_colptr[i][0]
  Scalar& diag_coeff(Index i) { return m_colptr[i][0]; }
  const Scalar& diag_coeff(Index i) const { return m_colptr[i][0]; }
  
  // Compute column pointers
  void compute_pointer()
  {
    m_colptr.clear();
    m_colptr.reserve(m_n);
    Scalar* head = m_data.data();
    
    for (Index i = 0; i < m_n; i++)
    {
      m_colptr.push_back(head);
      head += (m_n - i);
    }
  }
  
  // Copy mat - shift * I to m_data
  void copy_data(ConstGenericMatrix& mat, int uplo, const Scalar& shift)
  {
    if (uplo == Eigen::Lower)
    {
      for (Index j = 0; j < m_n; j++)
      {
        const Scalar* begin = &mat.coeffRef(j, j);
        const Index len = m_n - j;
        std::copy(begin, begin + len, col_pointer(j));
        diag_coeff(j) -= shift;
      }
    }
    else
    {
      Scalar* dest = m_data.data();
      for (Index i = 0; i < m_n; i++)
      {
        for (Index j = i; j < m_n; j++, dest++)
        {
          *dest = mat.coeff(i, j);
        }
        diag_coeff(i) -= shift;
      }
    }
  }
  
  // Compute compressed permutations
  void compress_permutation()
  {
    for (Index i = 0; i < m_n; i++)
    {
      // Recover the permutation action
      const Index perm = (m_perm[i] >= 0) ? (m_perm[i]) : (-m_perm[i] - 1);
      if (perm != i)
        m_permc.push_back(std::make_pair(i, perm));
    }
  }
  
  // Working on the A[k:end, k:end] submatrix
  // Exchange k <-> r
  // Assume r >= k
  void pivoting_1x1(Index k, Index r)
  {
    // No permutation
    if (k == r)
    {
      m_perm[k] = r;
      return;
    }
    
    // A[k, k] <-> A[r, r]
    std::swap(diag_coeff(k), diag_coeff(r));
    
    // A[(r+1):end, k] <-> A[(r+1):end, r]
    std::swap_ranges(&coeff(r + 1, k), col_pointer(k + 1), &coeff(r + 1, r));
    
    // A[(k+1):(r-1), k] <-> A[r, (k+1):(r-1)]
    Scalar* src = &coeff(k + 1, k);
    for (Index j = k + 1; j < r; j++, src++)
    {
      std::swap(*src, coeff(r, j));
    }
    
    m_perm[k] = r;
  }
  
  // Working on the A[k:end, k:end] submatrix
  // Exchange [k+1, k] <-> [r, p]
  // Assume p >= k, r >= k+1
  void pivoting_2x2(Index k, Index r, Index p)
  {
    pivoting_1x1(k, p);
    pivoting_1x1(k + 1, r);
    
    // A[k+1, k] <-> A[r, k]
    std::swap(coeff(k + 1, k), coeff(r, k));
    
    // Use negative signs to indicate a 2x2 block
    // Also minus one to distinguish a negative zero from a positive zero
    m_perm[k] = -m_perm[k] - 1;
    m_perm[k + 1] = -m_perm[k + 1] - 1;
  }
  
  // A[r1, c1:c2] <-> A[r2, c1:c2]
  // Assume r2 >= r1 > c2 >= c1
  void interchange_rows(Index r1, Index r2, Index c1, Index c2)
  {
    if (r1 == r2)
      return;
    
    for (Index j = c1; j <= c2; j++)
    {
      std::swap(coeff(r1, j), coeff(r2, j));
    }
  }
  
  // lambda = |A[r, k]| = max{|A[k+1, k]|, ..., |A[end, k]|}
  // Largest (in magnitude) off-diagonal element in the first column of the current reduced matrix
  // r is the row index
  // Assume k < end
  Scalar find_lambda(Index k, Index& r)
  {
    using std::abs;
    
    const Scalar* head = col_pointer(k);  // => A[k, k]
    const Scalar* end = col_pointer(k + 1);
    // Start with r=k+1, lambda=A[k+1, k]
    r = k + 1;
    Scalar lambda = abs(head[1]);
    // Scan remaining elements
    for (const Scalar* ptr = head + 2; ptr < end; ptr++)
    {
      const Scalar abs_elem = abs(*ptr);
      if (lambda < abs_elem)
      {
        lambda = abs_elem;
        r = k + (ptr - head);
      }
    }
    
    return lambda;
  }
  
  // sigma = |A[p, r]| = max {|A[k, r]|, ..., |A[end, r]|} \ {A[r, r]}
  // Largest (in magnitude) off-diagonal element in the r-th column of the current reduced matrix
  // p is the row index
  // Assume k < r < end
  Scalar find_sigma(Index k, Index r, Index& p)
  {
    using std::abs;
    
    // First search A[r+1, r], ...,  A[end, r], which has the same task as find_lambda()
    // If r == end, we skip this search
    Scalar sigma = Scalar(-1);
    if (r < m_n - 1)
      sigma = find_lambda(r, p);
    
    // Then search A[k, r], ..., A[r-1, r], which maps to A[r, k], ..., A[r, r-1]
    for (Index j = k; j < r; j++)
    {
      const Scalar abs_elem = abs(coeff(r, j));
      if (sigma < abs_elem)
      {
        sigma = abs_elem;
        p = j;
      }
    }
    
    return sigma;
  }
  
  // Generate permutations and apply to A
  // Return true if the resulting pivoting is 1x1, and false if 2x2
  bool permutate_mat(Index k, const Scalar& alpha)
  {
    using std::abs;
    
    Index r = k, p = k;
    const Scalar lambda = find_lambda(k, r);
    
    // If lambda=0, no need to interchange
    if (lambda > Scalar(0))
    {
      const Scalar abs_akk = abs(diag_coeff(k));
      // If |A[k, k]| >= alpha * lambda, no need to interchange
      if (abs_akk < alpha * lambda)
      {
        const Scalar sigma = find_sigma(k, r, p);
        
        // If sigma * |A[k, k]| >= alpha * lambda^2, no need to interchange
        if (sigma * abs_akk < alpha * lambda * lambda)
        {
          if (abs_akk >= alpha * sigma)
          {
            // Permutation on A
            pivoting_1x1(k, r);
            
            // Permutation on L
            interchange_rows(k, r, 0, k - 1);
            return true;
          }
          else
          {
            // There are two versions of permutation here
            // 1. A[k+1, k] <-> A[r, k]
            // 2. A[k+1, k] <-> A[r, p], where p >= k and r >= k+1
            //
            // Version 1 and 2 are used by Ref[1] and Ref[2], respectively
            
            // Version 1 implementation
            p = k;
            
            // Version 2 implementation
            // [r, p] and [p, r] are symmetric, but we need to make sure
            // p >= k and r >= k+1, so it is safe to always make r > p
            // One exception is when min{r,p} == k+1, in which case we make
            // r = k+1, so that only one permutation needs to be performed
            /* const Index rp_min = std::min(r, p);
             const Index rp_max = std::max(r, p);
             if(rp_min == k + 1)
             {
             r = rp_min; p = rp_max;
             } else {
             r = rp_max; p = rp_min;
             } */
            
            // Right now we use Version 1 since it reduces the overhead of interchange
            
            // Permutation on A
            pivoting_2x2(k, r, p);
            // Permutation on L
            interchange_rows(k, p, 0, k - 1);
            interchange_rows(k + 1, r, 0, k - 1);
            return false;
          }
        }
      }
    }
    
    return true;
  }
  
  // E = [e11, e12]
  //     [e21, e22]
  // Overwrite E with inv(E)
  void inverse_inplace_2x2(Scalar& e11, Scalar& e21, Scalar& e22) const
  {
    // inv(E) = [d11, d12], d11 = e22/delta, d21 = -e21/delta, d22 = e11/delta
    //          [d21, d22]
    const Scalar delta = e11 * e22 - e21 * e21;
    std::swap(e11, e22);
    e11 /= delta;
    e22 /= delta;
    e21 = -e21 / delta;
  }
  
  // Return value is the status, SUCCESSFUL/NUMERICAL_ISSUE
  int gaussian_elimination_1x1(Index k)
  {
    // D = 1 / A[k, k]
    const Scalar akk = diag_coeff(k);
    // Return NUMERICAL_ISSUE if not invertible
    if (akk == Scalar(0))
      return NUMERICAL_ISSUE;
    
    diag_coeff(k) = Scalar(1) / akk;
    
    // B -= l * l' / A[k, k], B := A[(k+1):end, (k+1):end], l := L[(k+1):end, k]
    Scalar* lptr = col_pointer(k) + 1;
    const Index ldim = m_n - k - 1;
    MapVec l(lptr, ldim);
    for (Index j = 0; j < ldim; j++)
    {
      MapVec(col_pointer(j + k + 1), ldim - j).noalias() -= (lptr[j] / akk) * l.tail(ldim - j);
    }
    
    // l /= A[k, k]
    l /= akk;
    
    return SUCCESSFUL;
  }
  
  // Return value is the status, SUCCESSFUL/NUMERICAL_ISSUE
  int gaussian_elimination_2x2(Index k)
  {
    // D = inv(E)
    Scalar& e11 = diag_coeff(k);
    Scalar& e21 = coeff(k + 1, k);
    Scalar& e22 = diag_coeff(k + 1);
    // Return NUMERICAL_ISSUE if not invertible
    if (e11 * e22 - e21 * e21 == Scalar(0))
      return NUMERICAL_ISSUE;
    
    inverse_inplace_2x2(e11, e21, e22);
    
    // X = l * inv(E), l := L[(k+2):end, k:(k+1)]
    Scalar* l1ptr = &coeff(k + 2, k);
    Scalar* l2ptr = &coeff(k + 2, k + 1);
    const Index ldim = m_n - k - 2;
    MapVec l1(l1ptr, ldim), l2(l2ptr, ldim);
    
    Eigen::Matrix<Scalar, Eigen::Dynamic, 2> X(ldim, 2);
    X.col(0).noalias() = l1 * e11 + l2 * e21;
    X.col(1).noalias() = l1 * e21 + l2 * e22;
    
    // B -= l * inv(E) * l' = X * l', B = A[(k+2):end, (k+2):end]
    for (Index j = 0; j < ldim; j++)
    {
      MapVec(col_pointer(j + k + 2), ldim - j).noalias() -= (X.col(0).tail(ldim - j) * l1ptr[j] + X.col(1).tail(ldim - j) * l2ptr[j]);
    }
    
    // l = X
    l1.noalias() = X.col(0);
    l2.noalias() = X.col(1);
    
    return SUCCESSFUL;
  }
  
public:
  BKLDLT() :
  m_n(0), m_computed(false), m_info(NOT_COMPUTED)
  {}
  
  // Factorize mat - shift * I
  BKLDLT(ConstGenericMatrix& mat, int uplo = Eigen::Lower, const Scalar& shift = Scalar(0)) :
  m_n(mat.rows()), m_computed(false), m_info(NOT_COMPUTED)
  {
    compute(mat, uplo, shift);
  }
  
  void compute(ConstGenericMatrix& mat, int uplo = Eigen::Lower, const Scalar& shift = Scalar(0))
  {
    using std::abs;
    
    m_n = mat.rows();
    if (m_n != mat.cols())
      throw std::invalid_argument("BKLDLT: matrix must be square");
    
    m_perm.setLinSpaced(m_n, 0, m_n - 1);
    m_permc.clear();
    
    // Copy data
    m_data.resize((m_n * (m_n + 1)) / 2);
    compute_pointer();
    copy_data(mat, uplo, shift);
    
    const Scalar alpha = (1.0 + std::sqrt(17.0)) / 8.0;
    Index k = 0;
    for (k = 0; k < m_n - 1; k++)
    {
      // 1. Interchange rows and columns of A, and save the result to m_perm
      bool is_1x1 = permutate_mat(k, alpha);
      
      // 2. Gaussian elimination
      if (is_1x1)
      {
        m_info = gaussian_elimination_1x1(k);
      }
      else
      {
        m_info = gaussian_elimination_2x2(k);
        k++;
      }
      
      // 3. Check status
      if (m_info != SUCCESSFUL)
        break;
    }
    // Invert the last 1x1 block if it exists
    if (k == m_n - 1)
    {
      const Scalar akk = diag_coeff(k);
      if (akk == Scalar(0))
        m_info = NUMERICAL_ISSUE;
      
      diag_coeff(k) = Scalar(1) / diag_coeff(k);
    }
    
    compress_permutation();
    
    m_computed = true;
  }
  
  // Solve Ax=b
  void solve_inplace(GenericVector b) const
  {
    if (!m_computed)
      throw std::logic_error("BKLDLT: need to call compute() first");
    
    // PAP' = LDL'
    // 1. b -> Pb
    Scalar* x = b.data();
    MapVec res(x, m_n);
    Index npermc = m_permc.size();
    for (Index i = 0; i < npermc; i++)
    {
      std::swap(x[m_permc[i].first], x[m_permc[i].second]);
    }
    
    // 2. Lz = Pb
    // If m_perm[end] < 0, then end with m_n - 3, otherwise end with m_n - 2
    const Index end = (m_perm[m_n - 1] < 0) ? (m_n - 3) : (m_n - 2);
    for (Index i = 0; i <= end; i++)
    {
      const Index b1size = m_n - i - 1;
      const Index b2size = b1size - 1;
      if (m_perm[i] >= 0)
      {
        MapConstVec l(&coeff(i + 1, i), b1size);
        res.segment(i + 1, b1size).noalias() -= l * x[i];
      }
      else
      {
        MapConstVec l1(&coeff(i + 2, i), b2size);
        MapConstVec l2(&coeff(i + 2, i + 1), b2size);
        res.segment(i + 2, b2size).noalias() -= (l1 * x[i] + l2 * x[i + 1]);
        i++;
      }
    }
    
    // 3. Dw = z
    for (Index i = 0; i < m_n; i++)
    {
      const Scalar e11 = diag_coeff(i);
      if (m_perm[i] >= 0)
      {
        x[i] *= e11;
      }
      else
      {
        const Scalar e21 = coeff(i + 1, i), e22 = diag_coeff(i + 1);
        const Scalar wi = x[i] * e11 + x[i + 1] * e21;
        x[i + 1] = x[i] * e21 + x[i + 1] * e22;
        x[i] = wi;
        i++;
      }
    }
    
    // 4. L'y = w
    // If m_perm[end] < 0, then start with m_n - 3, otherwise start with m_n - 2
    Index i = (m_perm[m_n - 1] < 0) ? (m_n - 3) : (m_n - 2);
    for (; i >= 0; i--)
    {
      const Index ldim = m_n - i - 1;
      MapConstVec l(&coeff(i + 1, i), ldim);
      x[i] -= res.segment(i + 1, ldim).dot(l);
      
      if (m_perm[i] < 0)
      {
        MapConstVec l2(&coeff(i + 1, i - 1), ldim);
        x[i - 1] -= res.segment(i + 1, ldim).dot(l2);
        i--;
      }
    }
    
    // 5. x = P'y
    for (i = npermc - 1; i >= 0; i--)
    {
      std::swap(x[m_permc[i].first], x[m_permc[i].second]);
    }
  }
  
  Vector solve(ConstGenericVector& b) const
  {
    Vector res = b;
    solve_inplace(res);
    return res;
  }
  
  int info() const { return m_info; }
};

}  // namespace LBFGSpp



namespace LBFGSpp {

//
// An *implicit* representation of the BFGS approximation to the Hessian matrix B
//
// B = theta * I - W * M * W'
// H = inv(B)
//
// Reference:
// [1] D. C. Liu and J. Nocedal (1989). On the limited memory BFGS method for large scale optimization.
// [2] R. H. Byrd, P. Lu, and J. Nocedal (1995). A limited memory algorithm for bound constrained optimization.
//
template <typename Scalar, bool LBFGSB = false>
class BFGSMat
{
private:
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using RefConstVec = Eigen::Ref<const Vector>;
  using IndexSet = std::vector<int>;
  
  int m_m;         // Maximum number of correction vectors
  Scalar m_theta;  // theta * I is the initial approximation to the Hessian matrix
  Matrix m_s;      // History of the s vectors
  Matrix m_y;      // History of the y vectors
  Vector m_ys;     // History of the s'y values
  Vector m_alpha;  // Temporary values used in computing H * v
  int m_ncorr;     // Number of correction vectors in the history, m_ncorr <= m
  int m_ptr;       // A Pointer to locate the most recent history, 1 <= m_ptr <= m
  // Details: s and y vectors are stored in cyclic order.
  //          For example, if the current s-vector is stored in m_s[, m-1],
  //          then in the next iteration m_s[, 0] will be overwritten.
  //          m_s[, m_ptr-1] points to the most recent history,
  //          and m_s[, m_ptr % m] points to the most distant one.
  
  //========== The following members are only used in L-BFGS-B algorithm ==========//
  Matrix m_permMinv;             // Permutated M inverse
  BKLDLT<Scalar> m_permMsolver;  // Represents the permutated M matrix
  
public:
  // Constructor
  BFGSMat() {}
  
  // Reset internal variables
  // n: dimension of the vector to be optimized
  // m: maximum number of corrections to approximate the Hessian matrix
  inline void reset(int n, int m)
  {
    m_m = m;
    m_theta = Scalar(1);
    m_s.resize(n, m);
    m_y.resize(n, m);
    m_ys.resize(m);
    m_alpha.resize(m);
    m_ncorr = 0;
    m_ptr = m;  // This makes sure that m_ptr % m == 0 in the first step
    
    if (LBFGSB)
    {
      m_permMinv.resize(2 * m, 2 * m);
      m_permMinv.setZero();
      m_permMinv.diagonal().setOnes();
    }
  }
  
  // Add correction vectors to the BFGS matrix
  inline void add_correction(const RefConstVec& s, const RefConstVec& y)
  {
    const int loc = m_ptr % m_m;
    
    m_s.col(loc).noalias() = s;
    m_y.col(loc).noalias() = y;
    
    // ys = y's = 1/rho
    const Scalar ys = m_s.col(loc).dot(m_y.col(loc));
    m_ys[loc] = ys;
    
    m_theta = m_y.col(loc).squaredNorm() / ys;
    
    if (m_ncorr < m_m)
      m_ncorr++;
    
    m_ptr = loc + 1;
    
    if (LBFGSB)
    {
      // Minv = [-D         L']
      //        [ L  theta*S'S]
      
      // Copy -D
      // Let S=[s[0], ..., s[m-1]], Y=[y[0], ..., y[m-1]]
      // D = [s[0]'y[0], ..., s[m-1]'y[m-1]]
      m_permMinv(loc, loc) = -ys;
      
      // Update S'S
      // We only store S'S in Minv, and multiply theta when LU decomposition is performed
      Vector Ss = m_s.leftCols(m_ncorr).transpose() * m_s.col(loc);
      m_permMinv.block(m_m + loc, m_m, 1, m_ncorr).noalias() = Ss.transpose();
      m_permMinv.block(m_m, m_m + loc, m_ncorr, 1).noalias() = Ss;
      
      // Compute L
      // L = [          0                                     ]
      //     [  s[1]'y[0]             0                       ]
      //     [  s[2]'y[0]     s[2]'y[1]                       ]
      //     ...
      //     [s[m-1]'y[0] ... ... ... ... ... s[m-1]'y[m-2]  0]
      //
      // L_next = [        0                                   ]
      //          [s[2]'y[1]             0                     ]
      //          [s[3]'y[1]     s[3]'y[2]                     ]
      //          ...
      //          [s[m]'y[1] ... ... ... ... ... s[m]'y[m-1]  0]
      const int len = m_ncorr - 1;
      // First zero out the column of oldest y
      if (m_ncorr >= m_m)
        m_permMinv.block(m_m, loc, m_m, 1).setZero();
      // Compute the row associated with new s
      // The current row is loc
      // End with column (loc + m - 1) % m
      // Length is len
      int yloc = (loc + m_m - 1) % m_m;
      for (int i = 0; i < len; i++)
      {
        m_permMinv(m_m + loc, yloc) = m_s.col(loc).dot(m_y.col(yloc));
        yloc = (yloc + m_m - 1) % m_m;
      }
      
      // Matrix LDLT factorization
      m_permMinv.block(m_m, m_m, m_m, m_m) *= m_theta;
      m_permMsolver.compute(m_permMinv);
      m_permMinv.block(m_m, m_m, m_m, m_m) /= m_theta;
    }
  }
  
  // Recursive formula to compute a * H * v, where a is a scalar, and v is [n x 1]
  // H0 = (1/theta) * I is the initial approximation to H
  // Algorithm 7.4 of Nocedal, J., & Wright, S. (2006). Numerical optimization.
  inline void apply_Hv(const Vector& v, const Scalar& a, Vector& res)
  {
    res.resize(v.size());
    
    // L-BFGS two-loop recursion
    
    // Loop 1
    res.noalias() = a * v;
    int j = m_ptr % m_m;
    for (int i = 0; i < m_ncorr; i++)
    {
      j = (j + m_m - 1) % m_m;
      m_alpha[j] = m_s.col(j).dot(res) / m_ys[j];
      res.noalias() -= m_alpha[j] * m_y.col(j);
    }
    
    // Apply initial H0
    res /= m_theta;
    
    // Loop 2
    for (int i = 0; i < m_ncorr; i++)
    {
      const Scalar beta = m_y.col(j).dot(res) / m_ys[j];
      res.noalias() += (m_alpha[j] - beta) * m_s.col(j);
      j = (j + 1) % m_m;
    }
  }
  
  //========== The following functions are only used in L-BFGS-B algorithm ==========//
  
  // Return the value of theta
  inline Scalar theta() const { return m_theta; }
  
  // Return current number of correction vectors
  inline int num_corrections() const { return m_ncorr; }
  
  // W = [Y, theta * S]
  // W [n x (2*ncorr)], v [n x 1], res [(2*ncorr) x 1]
  // res preserves the ordering of Y and S columns
  inline void apply_Wtv(const Vector& v, Vector& res) const
  {
    res.resize(2 * m_ncorr);
    res.head(m_ncorr).noalias() = m_y.leftCols(m_ncorr).transpose() * v;
    res.tail(m_ncorr).noalias() = m_theta * m_s.leftCols(m_ncorr).transpose() * v;
  }
  
  // The b-th row of the W matrix
  // Preserves the ordering of Y and S columns
  // Return as a column vector
  inline Vector Wb(int b) const
  {
    Vector res(2 * m_ncorr);
    for (int j = 0; j < m_ncorr; j++)
    {
      res[j] = m_y(b, j);
      res[m_ncorr + j] = m_s(b, j);
    }
    res.tail(m_ncorr) *= m_theta;
    return res;
  }
  
  // Extract rows of W
  inline Matrix Wb(const IndexSet& b) const
  {
    const int nb = b.size();
    const int* bptr = b.data();
    Matrix res(nb, 2 * m_ncorr);
    
    for (int j = 0; j < m_ncorr; j++)
    {
      const Scalar* Yptr = &m_y(0, j);
      const Scalar* Sptr = &m_s(0, j);
      Scalar* resYptr = res.data() + j * nb;
      Scalar* resSptr = resYptr + m_ncorr * nb;
      for (int i = 0; i < nb; i++)
      {
        const int row = bptr[i];
        resYptr[i] = Yptr[row];
        resSptr[i] = Sptr[row];
      }
    }
    return res;
  }
  
  // M is [(2*ncorr) x (2*ncorr)], v is [(2*ncorr) x 1]
  inline void apply_Mv(const Vector& v, Vector& res) const
  {
    res.resize(2 * m_ncorr);
    if (m_ncorr < 1)
      return;
    
    Vector vpadding = Vector::Zero(2 * m_m);
    vpadding.head(m_ncorr).noalias() = v.head(m_ncorr);
    vpadding.segment(m_m, m_ncorr).noalias() = v.tail(m_ncorr);
    
    // Solve linear equation
    m_permMsolver.solve_inplace(vpadding);
    
    res.head(m_ncorr).noalias() = vpadding.head(m_ncorr);
    res.tail(m_ncorr).noalias() = vpadding.segment(m_m, m_ncorr);
  }
  
  // Compute W'Pv
  // W [n x (2*ncorr)], v [nP x 1], res [(2*ncorr) x 1]
  // res preserves the ordering of Y and S columns
  // Returns false if the result is known to be zero
  inline bool apply_WtPv(const IndexSet& P_set, const Vector& v, Vector& res, bool test_zero = false) const
  {
    const int* Pptr = P_set.data();
    const Scalar* vptr = v.data();
    int nP = P_set.size();
    
    // Remove zeros in v to save computation
    IndexSet P_reduced;
    std::vector<Scalar> v_reduced;
    if (test_zero)
    {
      P_reduced.reserve(nP);
      for (int i = 0; i < nP; i++)
      {
        if (vptr[i] != Scalar(0))
        {
          P_reduced.push_back(Pptr[i]);
          v_reduced.push_back(vptr[i]);
        }
      }
      Pptr = P_reduced.data();
      vptr = v_reduced.data();
      nP = P_reduced.size();
    }
    
    res.resize(2 * m_ncorr);
    if (m_ncorr < 1 || nP < 1)
    {
      res.setZero();
      return false;
    }
    
    for (int j = 0; j < m_ncorr; j++)
    {
      Scalar resy = Scalar(0), ress = Scalar(0);
      const Scalar* yptr = &m_y(0, j);
      const Scalar* sptr = &m_s(0, j);
      for (int i = 0; i < nP; i++)
      {
        const int row = Pptr[i];
        resy += yptr[row] * vptr[i];
        ress += sptr[row] * vptr[i];
      }
      res[j] = resy;
      res[m_ncorr + j] = ress;
    }
    res.tail(m_ncorr) *= m_theta;
    return true;
  }
  
  // Compute s * P'WMv
  // Assume that v[2*ncorr x 1] has the same ordering (permutation) as W and M
  // Returns false if the result is known to be zero
  inline bool apply_PtWMv(const IndexSet& P_set, const Vector& v, Vector& res, const Scalar& scale) const
  {
    const int nP = P_set.size();
    res.resize(nP);
    res.setZero();
    if (m_ncorr < 1 || nP < 1)
      return false;
    
    Vector Mv;
    apply_Mv(v, Mv);
    // WP * Mv
    Mv.tail(m_ncorr) *= m_theta;
    for (int j = 0; j < m_ncorr; j++)
    {
      const Scalar* yptr = &m_y(0, j);
      const Scalar* sptr = &m_s(0, j);
      const Scalar Mvy = Mv[j], Mvs = Mv[m_ncorr + j];
      for (int i = 0; i < nP; i++)
      {
        const int row = P_set[i];
        res[i] += Mvy * yptr[row] + Mvs * sptr[row];
      }
    }
    res *= scale;
    return true;
  }
  // If the P'W matrix has been explicitly formed, do a direct matrix multiplication
  inline bool apply_PtWMv(const Matrix& WP, const Vector& v, Vector& res, const Scalar& scale) const
  {
    const int nP = WP.rows();
    res.resize(nP);
    if (m_ncorr < 1 || nP < 1)
    {
      res.setZero();
      return false;
    }
    
    Vector Mv;
    apply_Mv(v, Mv);
    // WP * Mv
    Mv.tail(m_ncorr) *= m_theta;
    res.noalias() = scale * (WP * Mv);
    return true;
  }
  
  // Compute F'BAb = -(F'W)M(W'AA'd)
  // W'd is known, and AA'+FF'=I, so W'AA'd = W'd - W'FF'd
  // Usually d contains many zeros, so we fist compute number of nonzero elements in A set and F set,
  // denoted as nnz_act and nnz_fv, respectively
  // If nnz_act is smaller, compute W'AA'd = WA' (A'd) directly
  // If nnz_fv is smaller, compute W'AA'd = W'd - WF' * (F'd)
  inline void compute_FtBAb(
      const Matrix& WF, const IndexSet& fv_set, const IndexSet& newact_set, const Vector& Wd, const Vector& drt,
      Vector& res) const
  {
    const int nact = newact_set.size();
    const int nfree = WF.rows();
    res.resize(nfree);
    if (m_ncorr < 1 || nact < 1 || nfree < 1)
    {
      res.setZero();
      return;
    }
    
    // W'AA'd
    Vector rhs(2 * m_ncorr);
    if (nact <= nfree)
    {
      // Construct A'd
      Vector Ad(nfree);
      for (int i = 0; i < nact; i++)
        Ad[i] = drt[newact_set[i]];
      apply_WtPv(newact_set, Ad, rhs);
    }
    else
    {
      // Construct F'd
      Vector Fd(nfree);
      for (int i = 0; i < nfree; i++)
        Fd[i] = drt[fv_set[i]];
      // Compute W'AA'd = W'd - WF' * (F'd)
      rhs.noalias() = WF.transpose() * Fd;
      rhs.tail(m_ncorr) *= m_theta;
      rhs.noalias() = Wd - rhs;
    }
    
    apply_PtWMv(WF, rhs, res, Scalar(-1));
  }
  
  // Compute inv(P'BP) * v
  // P represents an index set
  // inv(P'BP) * v = v / theta + WP * inv(inv(M) - WP' * WP / theta) * WP' * v / theta^2
  //
  // v is [nP x 1]
  inline void solve_PtBP(const Matrix& WP, const Vector& v, Vector& res) const
  {
    const int nP = WP.rows();
    res.resize(nP);
    if (m_ncorr < 1 || nP < 1)
    {
      res.noalias() = v / m_theta;
      return;
    }
    
    // Compute the matrix in the middle (only the lower triangular part is needed)
    // Remember that W = [Y, theta * S], but we do not store theta in WP
    Matrix mid(2 * m_ncorr, 2 * m_ncorr);
    // [0:(ncorr - 1), 0:(ncorr - 1)]
    for (int j = 0; j < m_ncorr; j++)
    {
      mid.col(j).segment(j, m_ncorr - j).noalias() = m_permMinv.col(j).segment(j, m_ncorr - j) -
        WP.block(0, j, nP, m_ncorr - j).transpose() * WP.col(j) / m_theta;
    }
    // [ncorr:(2 * ncorr - 1), 0:(ncorr - 1)]
    mid.block(m_ncorr, 0, m_ncorr, m_ncorr).noalias() = m_permMinv.block(m_m, 0, m_ncorr, m_ncorr) -
      WP.rightCols(m_ncorr).transpose() * WP.leftCols(m_ncorr);
    // [ncorr:(2 * ncorr - 1), ncorr:(2 * ncorr - 1)]
    for (int j = 0; j < m_ncorr; j++)
    {
      mid.col(m_ncorr + j).segment(m_ncorr + j, m_ncorr - j).noalias() = m_theta *
        (m_permMinv.col(m_m + j).segment(m_m + j, m_ncorr - j) - WP.rightCols(m_ncorr - j).transpose() * WP.col(m_ncorr + j));
    }
    // Factorization
    BKLDLT<Scalar> midsolver(mid);
    // Compute the final result
    Vector WPv = WP.transpose() * v;
    WPv.tail(m_ncorr) *= m_theta;
    midsolver.solve_inplace(WPv);
    WPv.tail(m_ncorr) *= m_theta;
    res.noalias() = v / m_theta + (WP * WPv) / (m_theta * m_theta);
  }
  
  // Compute P'BQv, where P and Q are two mutually exclusive index selection operators
  // P'BQv = -WP * M * WQ' * v
  // Returns false if the result is known to be zero
  inline bool apply_PtBQv(const Matrix& WP, const IndexSet& Q_set, const Vector& v, Vector& res, bool test_zero = false) const
  {
    const int nP = WP.rows();
    const int nQ = Q_set.size();
    res.resize(nP);
    if (m_ncorr < 1 || nP < 1 || nQ < 1)
    {
      res.setZero();
      return false;
    }
    
    Vector WQtv;
    bool nonzero = apply_WtPv(Q_set, v, WQtv, test_zero);
    if (!nonzero)
    {
      res.setZero();
      return false;
    }
    
    Vector MWQtv;
    apply_Mv(WQtv, MWQtv);
    MWQtv.tail(m_ncorr) *= m_theta;
    res.noalias() = -WP * MWQtv;
    return true;
  }
  // If the Q'W matrix has been explicitly formed, do a direct matrix multiplication
  inline bool apply_PtBQv(const Matrix& WP, const Matrix& WQ, const Vector& v, Vector& res) const
  {
    const int nP = WP.rows();
    const int nQ = WQ.rows();
    res.resize(nP);
    if (m_ncorr < 1 || nP < 1 || nQ < 1)
    {
      res.setZero();
      return false;
    }
    
    // Remember that W = [Y, theta * S], so we need to multiply theta to the second half
    Vector WQtv = WQ.transpose() * v;
    WQtv.tail(m_ncorr) *= m_theta;
    Vector MWQtv;
    apply_Mv(WQtv, MWQtv);
    MWQtv.tail(m_ncorr) *= m_theta;
    res.noalias() = -WP * MWQtv;
    return true;
  }
};

}  // namespace LBFGSpp


namespace LBFGSpp {

//
// Class to compute the generalized Cauchy point (GCP) for the L-BFGS-B algorithm,
// mainly for internal use.
//
// The target of the GCP procedure is to find a step size t such that
// x(t) = x0 - t * g is a local minimum of the quadratic function m(x),
// where m(x) is a local approximation to the objective function.
//
// First determine a sequence of break points t0=0, t1, t2, ..., tn.
// On each interval [t[i-1], t[i]], x is changing linearly.
// After passing a break point, one or more coordinates of x will be fixed at the bounds.
// We search the first local minimum of m(x) by examining the intervals [t[i-1], t[i]] sequentially.
//
// Reference:
// [1] R. H. Byrd, P. Lu, and J. Nocedal (1995). A limited memory algorithm for bound constrained optimization.
//
template <typename Scalar>
class ArgSort
{
private:
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using IndexSet = std::vector<int>;
  
  const Scalar* values;
  
public:
  ArgSort(const Vector& value_vec) :
  values(value_vec.data())
  {}
  
  inline bool operator()(int key1, int key2) { return values[key1] < values[key2]; }
  inline void sort_key(IndexSet& key_vec) const
  {
    std::sort(key_vec.begin(), key_vec.end(), *this);
  }
};

template <typename Scalar>
class Cauchy
{
private:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  typedef Eigen::Matrix<int, Eigen::Dynamic, 1> IntVector;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
  typedef std::vector<int> IndexSet;
  
  // Find the smallest index i such that brk[ord[i]] > t, assuming brk[ord] is already sorted.
  // If the return value equals n, then all values are <= t.
  static int search_greater(const Vector& brk, const IndexSet& ord, const Scalar& t, int start = 0)
  {
    const int nord = ord.size();
    int i;
    for (i = start; i < nord; i++)
    {
      if (brk[ord[i]] > t)
        break;
    }
    
    return i;
  }
  
public:
  // bfgs:       An object that represents the BFGS approximation matrix.
  // x0:         Current parameter vector.
  // g:          Gradient at x0.
  // lb:         Lower bounds for x.
  // ub:         Upper bounds for x.
  // xcp:        The output generalized Cauchy point.
  // vecc:       c = W'(xcp - x0), used in the subspace minimization routine.
  // newact_set: Coordinates that newly become active during the GCP procedure.
  // fv_set:     Free variable set.
  static void get_cauchy_point(
      const BFGSMat<Scalar, true>& bfgs, const Vector& x0, const Vector& g, const Vector& lb, const Vector& ub,
      Vector& xcp, Vector& vecc, IndexSet& newact_set, IndexSet& fv_set)
  {
    // std::cout << "========================= Entering GCP search =========================\n\n";
    
    // Initialization
    const int n = x0.size();
    xcp.resize(n);
    xcp.noalias() = x0;
    vecc.resize(2 * bfgs.num_corrections());
    vecc.setZero();
    newact_set.clear();
    newact_set.reserve(n);
    fv_set.clear();
    fv_set.reserve(n);
    
    // Construct break points
    Vector brk(n), vecd(n);
    // If brk[i] == 0, i belongs to active set
    // If brk[i] == Inf, i belongs to free variable set
    // Others are currently undecided
    IndexSet ord;
    ord.reserve(n);
    const Scalar inf = std::numeric_limits<Scalar>::infinity();
    for (int i = 0; i < n; i++)
    {
      if (lb[i] == ub[i])
        brk[i] = Scalar(0);
      else if (g[i] < Scalar(0))
        brk[i] = (x0[i] - ub[i]) / g[i];
      else if (g[i] > Scalar(0))
        brk[i] = (x0[i] - lb[i]) / g[i];
      else
        brk[i] = inf;
      
      const bool iszero = (brk[i] == Scalar(0));
      vecd[i] = iszero ? Scalar(0) : -g[i];
      
      if (brk[i] == inf)
        fv_set.push_back(i);
      else if (!iszero)
        ord.push_back(i);
    }
    
    // Sort indices of break points
    ArgSort<Scalar> sorting(brk);
    sorting.sort_key(ord);
    
    // Break points `brko := brk[ord]` are in increasing order
    // `ord` contains the coordinates that define the corresponding break points
    // brk[i] == 0 <=> The i-th coordinate is on the boundary
    const int nord = ord.size();
    const int nfree = fv_set.size();
    if ((nfree < 1) && (nord < 1))
    {
      /* std::cout << "** All coordinates at boundary **\n";
       std::cout << "\n========================= Leaving GCP search =========================\n\n"; */
      return;
    }
    
    // First interval: [il=0, iu=brk[ord[0]]]
    // In case ord is empty, we take iu=Inf
    
    // p = W'd, c = 0
    Vector vecp;
    bfgs.apply_Wtv(vecd, vecp);
    // f' = -d'd
    Scalar fp = -vecd.squaredNorm();
    // f'' = -theta * f' - p'Mp
    Vector cache;
    bfgs.apply_Mv(vecp, cache);  // cache = Mp
    Scalar fpp = -bfgs.theta() * fp - vecp.dot(cache);
    
    // Theoretical step size to move
    Scalar deltatmin = -fp / fpp;
    
    // Limit on the current interval
    Scalar il = Scalar(0);
    // We have excluded the case that max(brk) <= 0
    int b = 0;
    Scalar iu = (nord < 1) ? inf : brk[ord[b]];
    Scalar deltat = iu - il;
    
    /* int iter = 0;
     std::cout << "** Iter " << iter << " **\n";
     std::cout << "   fp = " << fp << ", fpp = " << fpp << ", deltatmin = " << deltatmin << std::endl;
     std::cout << "   il = " << il << ", iu = " << iu << ", deltat = " << deltat << std::endl; */
    
    // If deltatmin >= deltat, we need to do the following things:
    // 1. Update vecc
    // 2. Since we are going to cross iu, the coordinates that define iu become active
    // 3. Update some quantities on these new active coordinates (xcp, vecd, vecp)
    // 4. Move to the next interval and compute the new deltatmin
    bool crossed_all = false;
    const int ncorr = bfgs.num_corrections();
    Vector wact(2 * ncorr);
    while (deltatmin >= deltat)
    {
      // Step 1
      vecc.noalias() += deltat * vecp;
      
      // Step 2
      // First check how many coordinates will be active when we cross the previous iu
      // b is the smallest number such that brko[b] == iu
      // Let bp be the largest number such that brko[bp] == iu
      // Then coordinates ord[b] to ord[bp] will be active
      const int act_begin = b;
      const int act_end = search_greater(brk, ord, iu, b) - 1;
      
      // If nfree == 0 and act_end == nord-1, then we have crossed all coordinates
      // We only need to update xcp from ord[b] to ord[bp], and then exit
      if ((nfree == 0) && (act_end == nord - 1))
      {
        // std::cout << "** [ ";
        for (int i = act_begin; i <= act_end; i++)
        {
          const int act = ord[i];
          xcp[act] = (vecd[act] > Scalar(0)) ? ub[act] : lb[act];
          newact_set.push_back(act);
          // std::cout << act + 1 << " ";
        }
        // std::cout << "] become active **\n\n";
        // std::cout << "** All break points visited **\n\n";
        
        crossed_all = true;
        break;
      }
      
      // Step 3
      // Update xcp and d on active coordinates
      // std::cout << "** [ ";
      fp += deltat * fpp;
      for (int i = act_begin; i <= act_end; i++)
      {
        const int act = ord[i];
        xcp[act] = (vecd[act] > Scalar(0)) ? ub[act] : lb[act];
        // z = xcp - x0
        const Scalar zact = xcp[act] - x0[act];
        const Scalar gact = g[act];
        const Scalar ggact = gact * gact;
        wact.noalias() = bfgs.Wb(act);
        bfgs.apply_Mv(wact, cache);  // cache = Mw
        fp += ggact + bfgs.theta() * gact * zact - gact * cache.dot(vecc);
        fpp -= (bfgs.theta() * ggact + 2 * gact * cache.dot(vecp) + ggact * cache.dot(wact));
        vecp.noalias() += gact * wact;
        vecd[act] = Scalar(0);
        newact_set.push_back(act);
        // std::cout << act + 1 << " ";
      }
      // std::cout << "] become active **\n\n";
      
      // Step 4
      // Theoretical step size to move
      deltatmin = -fp / fpp;
      // Update interval bound
      il = iu;
      b = act_end + 1;
      // If we have visited all finite-valued break points, and have not exited earlier,
      // then the next iu will be infinity. Simply exit the loop now
      if (b >= nord)
        break;
      iu = brk[ord[b]];
      // Width of the current interval
      deltat = iu - il;
      
      /* iter++;
       std::cout << "** Iter " << iter << " **\n";
       std::cout << "   fp = " << fp << ", fpp = " << fpp << ", deltatmin = " << deltatmin << std::endl;
       std::cout << "   il = " << il << ", iu = " << iu << ", deltat = " << deltat << std::endl; */
    }
    
    // In some rare cases fpp is numerically zero, making deltatmin equal to Inf
    // If this happens, force fpp to be the machine precision
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();
    if (fpp < eps)
      deltatmin = -fp / eps;
    
    // Last step
    if (!crossed_all)
    {
      deltatmin = std::max(deltatmin, Scalar(0));
      vecc.noalias() += deltatmin * vecp;
      const Scalar tfinal = il + deltatmin;
      // Update xcp on free variable coordinates
      for (int i = 0; i < nfree; i++)
      {
        const int coord = fv_set[i];
        xcp[coord] = x0[coord] + tfinal * vecd[coord];
      }
      for (int i = b; i < nord; i++)
      {
        const int coord = ord[i];
        xcp[coord] = x0[coord] + tfinal * vecd[coord];
        fv_set.push_back(coord);
      }
    }
    // std::cout << "\n========================= Leaving GCP search =========================\n\n";
  }
};

}  // namespace LBFGSpp


namespace LBFGSpp {

//
// Subspace minimization procedure of the L-BFGS-B algorithm,
// mainly for internal use.
//
// The target of subspace minimization is to minimize the quadratic function m(x)
// over the free variables, subject to the bound condition.
// Free variables stand for coordinates that are not at the boundary in xcp,
// the generalized Cauchy point.
//
// In the classical implementation of L-BFGS-B [1], the minimization is done by first
// ignoring the box constraints, followed by a line search. Our implementation is
// an exact minimization subject to the bounds, based on the BOXCQP algorithm [2].
//
// Reference:
// [1] R. H. Byrd, P. Lu, and J. Nocedal (1995). A limited memory algorithm for bound constrained optimization.
// [2] C. Voglis and I. E. Lagaris (2004). BOXCQP: An algorithm for bound constrained convex quadratic problems.
//
template <typename Scalar>
class SubspaceMin
{
private:
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using IndexSet = std::vector<int>;
  
  // v[ind]
  static Vector subvec(const Vector& v, const IndexSet& ind)
  {
    const int nsub = ind.size();
    Vector res(nsub);
    for (int i = 0; i < nsub; i++)
      res[i] = v[ind[i]];
    return res;
  }
  
  // v[ind] = rhs
  static void subvec_assign(Vector& v, const IndexSet& ind, const Vector& rhs)
  {
    const int nsub = ind.size();
    for (int i = 0; i < nsub; i++)
      v[ind[i]] = rhs[i];
  }
  
  // Check whether the vector is within the bounds
  static bool in_bounds(const Vector& x, const Vector& lb, const Vector& ub)
  {
    const int n = x.size();
    for (int i = 0; i < n; i++)
    {
      if (x[i] < lb[i] || x[i] > ub[i])
        return false;
    }
    return true;
  }
  
  // Test convergence of P set
  static bool P_converged(const IndexSet& yP_set, const Vector& vecy, const Vector& vecl, const Vector& vecu)
  {
    const int nP = yP_set.size();
    for (int i = 0; i < nP; i++)
    {
      const int coord = yP_set[i];
      if (vecy[coord] < vecl[coord] || vecy[coord] > vecu[coord])
        return false;
    }
    return true;
  }
  
  // Test convergence of L set
  static bool L_converged(const IndexSet& yL_set, const Vector& lambda)
  {
    const int nL = yL_set.size();
    for (int i = 0; i < nL; i++)
    {
      const int coord = yL_set[i];
      if (lambda[coord] < Scalar(0))
        return false;
    }
    return true;
  }
  
  // Test convergence of L set
  static bool U_converged(const IndexSet& yU_set, const Vector& mu)
  {
    const int nU = yU_set.size();
    for (int i = 0; i < nU; i++)
    {
      const int coord = yU_set[i];
      if (mu[coord] < Scalar(0))
        return false;
    }
    return true;
  }
  
public:
  // bfgs:       An object that represents the BFGS approximation matrix.
  // x0:         Current parameter vector.
  // xcp:        Computed generalized Cauchy point.
  // g:          Gradient at x0.
  // lb:         Lower bounds for x.
  // ub:         Upper bounds for x.
  // Wd:         W'(xcp - x0)
  // newact_set: Coordinates that newly become active during the GCP procedure.
  // fv_set:     Free variable set.
  // maxit:      Maximum number of iterations.
  // drt:        The output direction vector, drt = xsm - x0.
  static void subspace_minimize(
      const BFGSMat<Scalar, true>& bfgs, const Vector& x0, const Vector& xcp, const Vector& g,
      const Vector& lb, const Vector& ub, const Vector& Wd, const IndexSet& newact_set, const IndexSet& fv_set, int maxit,
      Vector& drt)
  {
    // std::cout << "========================= Entering subspace minimization =========================\n\n";
    
    // d = xcp - x0
    drt.noalias() = xcp - x0;
    // Size of free variables
    const int nfree = fv_set.size();
    // If there is no free variable, simply return drt
    if (nfree < 1)
    {
      // std::cout << "========================= (Early) leaving subspace minimization =========================\n\n";
      return;
    }
    
    // std::cout << "New active set = [ "; for(std::size_t i = 0; i < newact_set.size(); i++)  std::cout << newact_set[i] << " "; std::cout << "]\n";
    // std::cout << "Free variable set = [ "; for(std::size_t i = 0; i < fv_set.size(); i++)  std::cout << fv_set[i] << " "; std::cout << "]\n\n";
    
    // Extract the rows of W in the free variable set
    Matrix WF = bfgs.Wb(fv_set);
    // Compute F'BAb = -F'WMW'AA'd
    Vector vecc(nfree);
    bfgs.compute_FtBAb(WF, fv_set, newact_set, Wd, drt, vecc);
    // Set the vector c=F'BAb+F'g for linear term, and vectors l and u for the new bounds
    Vector vecl(nfree), vecu(nfree);
    for (int i = 0; i < nfree; i++)
    {
      const int coord = fv_set[i];
      vecl[i] = lb[coord] - x0[coord];
      vecu[i] = ub[coord] - x0[coord];
      vecc[i] += g[coord];
    }
    // Solve y = -inv(B[F, F]) * c
    Vector vecy(nfree);
    bfgs.solve_PtBP(WF, -vecc, vecy);
    // Test feasibility
    // If yes, then the solution has been found
    if (in_bounds(vecy, vecl, vecu))
    {
      subvec_assign(drt, fv_set, vecy);
      return;
    }
    // Otherwise, enter the iterations
    
    // Make a copy of y as a fallback solution
    Vector yfallback = vecy;
    // Dual variables
    Vector lambda = Vector::Zero(nfree), mu = Vector::Zero(nfree);
    
    // Iterations
    IndexSet L_set, U_set, P_set, yL_set, yU_set, yP_set;
    L_set.reserve(nfree / 3);
    yL_set.reserve(nfree / 3);
    U_set.reserve(nfree / 3);
    yU_set.reserve(nfree / 3);
    P_set.reserve(nfree);
    yP_set.reserve(nfree);
    int k;
    for (k = 0; k < maxit; k++)
    {
      // Construct the L, U, and P sets, and then update values
      // Indices in original drt vector
      L_set.clear();
      U_set.clear();
      P_set.clear();
      // Indices in y
      yL_set.clear();
      yU_set.clear();
      yP_set.clear();
      for (int i = 0; i < nfree; i++)
      {
        const int coord = fv_set[i];
        const Scalar li = vecl[i], ui = vecu[i];
        if ((vecy[i] < li) || (vecy[i] == li && lambda[i] >= Scalar(0)))
        {
          L_set.push_back(coord);
          yL_set.push_back(i);
          vecy[i] = li;
          mu[i] = Scalar(0);
        }
        else if ((vecy[i] > ui) || (vecy[i] == ui && mu[i] >= Scalar(0)))
        {
          U_set.push_back(coord);
          yU_set.push_back(i);
          vecy[i] = ui;
          lambda[i] = Scalar(0);
        }
        else
        {
          P_set.push_back(coord);
          yP_set.push_back(i);
          lambda[i] = Scalar(0);
          mu[i] = Scalar(0);
        }
      }
      
      /* std::cout << "** Iter " << k << " **\n";
       std::cout << "   L = [ "; for(std::size_t i = 0; i < L_set.size(); i++)  std::cout << L_set[i] << " "; std::cout << "]\n";
       std::cout << "   U = [ "; for(std::size_t i = 0; i < U_set.size(); i++)  std::cout << U_set[i] << " "; std::cout << "]\n";
       std::cout << "   P = [ "; for(std::size_t i = 0; i < P_set.size(); i++)  std::cout << P_set[i] << " "; std::cout << "]\n\n"; */
      
      // Extract the rows of W in the P set
      Matrix WP = bfgs.Wb(P_set);
      // Solve y[P] = -inv(B[P, P]) * (B[P, L] * l[L] + B[P, U] * u[U] + c[P])
      const int nP = P_set.size();
      if (nP > 0)
      {
        Vector rhs = subvec(vecc, yP_set);
        Vector lL = subvec(vecl, yL_set);
        Vector uU = subvec(vecu, yU_set);
        Vector tmp(nP);
        bool nonzero = bfgs.apply_PtBQv(WP, L_set, lL, tmp, true);
        if (nonzero)
          rhs.noalias() += tmp;
        nonzero = bfgs.apply_PtBQv(WP, U_set, uU, tmp, true);
        if (nonzero)
          rhs.noalias() += tmp;
        
        bfgs.solve_PtBP(WP, -rhs, tmp);
        subvec_assign(vecy, yP_set, tmp);
      }
      
      // Solve lambda[L] = B[L, F] * y + c[L]
      const int nL = L_set.size();
      const int nU = U_set.size();
      Vector Fy;
      if (nL > 0 || nU > 0)
        bfgs.apply_WtPv(fv_set, vecy, Fy);
      if (nL > 0)
      {
        Vector res;
        bfgs.apply_PtWMv(L_set, Fy, res, Scalar(-1));
        res.noalias() += subvec(vecc, yL_set) + bfgs.theta() * subvec(vecy, yL_set);
        subvec_assign(lambda, yL_set, res);
      }
      
      // Solve mu[U] = -B[U, F] * y - c[U]
      if (nU > 0)
      {
        Vector negRes;
        bfgs.apply_PtWMv(U_set, Fy, negRes, Scalar(-1));
        negRes.noalias() += subvec(vecc, yU_set) + bfgs.theta() * subvec(vecy, yU_set);
        subvec_assign(mu, yU_set, -negRes);
      }
      
      // Test convergence
      if (L_converged(yL_set, lambda) && U_converged(yU_set, mu) && P_converged(yP_set, vecy, vecl, vecu))
        break;
    }
    
    // If the iterations do not converge, try the projection
    if (k >= maxit)
    {
      vecy.noalias() = vecy.cwiseMax(vecl).cwiseMin(vecu);
      subvec_assign(drt, fv_set, vecy);
      // Test whether drt is a descent direction
      Scalar dg = drt.dot(g);
      // If yes, return the result
      if (dg <= -std::numeric_limits<Scalar>::epsilon())
        return;
      
      // If not, fall back to the projected unconstrained solution
      vecy.noalias() = yfallback.cwiseMax(vecl).cwiseMin(vecu);
      subvec_assign(drt, fv_set, vecy);
      dg = drt.dot(g);
      if (dg <= -std::numeric_limits<Scalar>::epsilon())
        return;
      
      // If still not, fall back to the unconstrained solution
      subvec_assign(drt, fv_set, yfallback);
      return;
    }
    
    // std::cout << "** Minimization finished in " << k + 1 << " iteration(s) **\n\n";
    // std::cout << "========================= Leaving subspace minimization =========================\n\n";
    
    subvec_assign(drt, fv_set, vecy);
  }
};

}  // namespace LBFGSpp


namespace LBFGSpp {

///
/// The line search algorithm by Moré and Thuente (1994), currently used for the L-BFGS-B algorithm.
///
/// The target of this line search algorithm is to find a step size \f$\alpha\f$ that satisfies the strong Wolfe condition
/// \f$f(x+\alpha d) \le f(x) + \alpha\mu g(x)^T d\f$ and \f$|g(x+\alpha d)^T d| \le \eta|g(x)^T d|\f$.
/// Our implementation is a simplified version of the algorithm in [1]. We assume that \f$0<\mu<\eta<1\f$, while in [1]
/// they do not assume \f$\eta>\mu\f$. As a result, the algorithm in [1] has two stages, but in our implementation we
/// only need the first stage to guarantee the convergence.
///
/// Reference:
/// [1] Moré, J. J., & Thuente, D. J. (1994). Line search algorithms with guaranteed sufficient decrease.
///
template <typename Scalar>
class LineSearchMoreThuente
{
private:
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  
  // Minimizer of a quadratic function q(x) = c0 + c1 * x + c2 * x^2
  // that interpolates fa, ga, and fb, assuming the minimizer exists
  // For case I: fb >= fa and ga * (b - a) < 0
  static Scalar quadratic_minimizer(const Scalar& a, const Scalar& b, const Scalar& fa, const Scalar& ga, const Scalar& fb)
  {
    const Scalar ba = b - a;
    const Scalar w = Scalar(0.5) * ba * ga / (fa - fb + ba * ga);
    return a + w * ba;
  }
  
  // Minimizer of a quadratic function q(x) = c0 + c1 * x + c2 * x^2
  // that interpolates fa, ga and gb, assuming the minimizer exists
  // The result actually does not depend on fa
  // For case II: ga * (b - a) < 0, ga * gb < 0
  // For case III: ga * (b - a) < 0, ga * ga >= 0, |gb| <= |ga|
  static Scalar quadratic_minimizer(const Scalar& a, const Scalar& b, const Scalar& ga, const Scalar& gb)
  {
    const Scalar w = ga / (ga - gb);
    return a + w * (b - a);
  }
  
  // Local minimizer of a cubic function q(x) = c0 + c1 * x + c2 * x^2 + c3 * x^3
  // that interpolates fa, ga, fb and gb, assuming a != b
  // Also sets a flag indicating whether the minimizer exists
  static Scalar cubic_minimizer(const Scalar& a, const Scalar& b, const Scalar& fa, const Scalar& fb,
                                const Scalar& ga, const Scalar& gb, bool& exists)
  {
    using std::abs;
    using std::sqrt;
    
    const Scalar apb = a + b;
    const Scalar ba = b - a;
    const Scalar ba2 = ba * ba;
    const Scalar fba = fb - fa;
    const Scalar gba = gb - ga;
    // z3 = c3 * (b-a)^3, z2 = c2 * (b-a)^3, z1 = c1 * (b-a)^3
    const Scalar z3 = (ga + gb) * ba - Scalar(2) * fba;
    const Scalar z2 = Scalar(0.5) * (gba * ba2 - Scalar(3) * apb * z3);
    const Scalar z1 = fba * ba2 - apb * z2 - (a * apb + b * b) * z3;
    // std::cout << "z1 = " << z1 << ", z2 = " << z2 << ", z3 = " << z3 << std::endl;
    
    // If c3 = z/(b-a)^3 == 0, reduce to quadratic problem
    const Scalar eps = std::numeric_limits<Scalar>::epsilon();
    if (abs(z3) < eps * abs(z2) || abs(z3) < eps * abs(z1))
    {
      // Minimizer exists if c2 > 0
      exists = (z2 * ba > Scalar(0));
      // Return the end point if the minimizer does not exist
      return exists ? (-Scalar(0.5) * z1 / z2) : b;
    }
    
    // Now we can assume z3 > 0
    // The minimizer is a solution to the equation c1 + 2*c2 * x + 3*c3 * x^2 = 0
    // roots = -(z2/z3) / 3 (+-) sqrt((z2/z3)^2 - 3 * (z1/z3)) / 3
    //
    // Let u = z2/(3z3) and v = z1/z2
    // The minimizer exists if v/u <= 1
    const Scalar u = z2 / (Scalar(3) * z3), v = z1 / z2;
    const Scalar vu = v / u;
    exists = (vu <= Scalar(1));
    if (!exists)
      return b;
    
    // We need to find a numerically stable way to compute the roots, as z3 may still be small
    //
    // If |u| >= |v|, let w = 1 + sqrt(1-v/u), and then
    // r1 = -u * w, r2 = -v / w, r1 does not need to be the smaller one
    //
    // If |u| < |v|, we must have uv <= 0, and then
    // r = -u (+-) sqrt(delta), where
    // sqrt(delta) = sqrt(|u|) * sqrt(|v|) * sqrt(1-u/v)
    Scalar r1 = Scalar(0), r2 = Scalar(0);
    if (abs(u) >= abs(v))
    {
      const Scalar w = Scalar(1) + sqrt(Scalar(1) - vu);
      r1 = -u * w;
      r2 = -v / w;
    }
    else
    {
      const Scalar sqrtd = sqrt(abs(u)) * sqrt(abs(v)) * sqrt(1 - u / v);
      r1 = -u - sqrtd;
      r2 = -u + sqrtd;
    }
    return (z3 * ba > Scalar(0)) ? ((std::max)(r1, r2)) : ((std::min)(r1, r2));
  }
  
  // Select the next step size according to the current step sizes,
  // function values, and derivatives
  static Scalar step_selection(
      const Scalar& al, const Scalar& au, const Scalar& at,
      const Scalar& fl, const Scalar& fu, const Scalar& ft,
      const Scalar& gl, const Scalar& gu, const Scalar& gt)
  {
    using std::abs;
    
    if (al == au)
      return al;
    
    // If ft = Inf or gt = Inf, we return the middle point of al and at
    if (!std::isfinite(ft) || !std::isfinite(gt))
      return (al + at) / Scalar(2);
    
    // ac: cubic interpolation of fl, ft, gl, gt
    // aq: quadratic interpolation of fl, gl, ft
    bool ac_exists;
    // std::cout << "al = " << al << ", at = " << at << ", fl = " << fl << ", ft = " << ft << ", gl = " << gl << ", gt = " << gt << std::endl;
    const Scalar ac = cubic_minimizer(al, at, fl, ft, gl, gt, ac_exists);
    const Scalar aq = quadratic_minimizer(al, at, fl, gl, ft);
    // std::cout << "ac = " << ac << ", aq = " << aq << std::endl;
    // Case 1: ft > fl
    if (ft > fl)
    {
      // This should not happen if ft > fl, but just to be safe
      if (!ac_exists)
        return aq;
      // Then use the scheme described in the paper
      return (abs(ac - al) < abs(aq - al)) ? ac : ((aq + ac) / Scalar(2));
    }
    
    // as: quadratic interpolation of gl and gt
    const Scalar as = quadratic_minimizer(al, at, gl, gt);
    // Case 2: ft <= fl, gt * gl < 0
    if (gt * gl < Scalar(0))
      return (abs(ac - at) >= abs(as - at)) ? ac : as;
    
    // Case 3: ft <= fl, gt * gl >= 0, |gt| < |gl|
    const Scalar deltal = Scalar(1.1), deltau = Scalar(0.66);
    if (abs(gt) < abs(gl))
    {
      // We choose either ac or as
      // The case for ac: 1. It exists, and
      //                  2. ac is farther than at from al, and
      //                  3. ac is closer to at than as
      // Cases for as: otherwise
      const Scalar res = (ac_exists &&
                          (ac - at) * (at - al) > Scalar(0) &&
                          abs(ac - at) < abs(as - at)) ?
                          ac :
      as;
      // Postprocessing the chosen step
      return (at > al) ?
      std::min(at + deltau * (au - at), res) :
        std::max(at + deltau * (au - at), res);
    }
    
    // Simple extrapolation if au, fu, or gu is infinity
    if ((!std::isfinite(au)) || (!std::isfinite(fu)) || (!std::isfinite(gu)))
      return at + deltal * (at - al);
    
    // ae: cubic interpolation of ft, fu, gt, gu
    bool ae_exists;
    const Scalar ae = cubic_minimizer(at, au, ft, fu, gt, gu, ae_exists);
    // Case 4: ft <= fl, gt * gl >= 0, |gt| >= |gl|
    // The following is not used in the paper, but it seems to be a reasonable safeguard
    return (at > al) ?
    std::min(at + deltau * (au - at), ae) :
      std::max(at + deltau * (au - at), ae);
  }
  
public:
  ///
  /// Line search by Moré and Thuente (1994).
  ///
  /// \param f        A function object such that `f(x, grad)` returns the
  ///                 objective function value at `x`, and overwrites `grad` with
  ///                 the gradient.
  /// \param param    An `LBFGSParam` or `LBFGSBParam` object that stores the
  ///                 parameters of the solver.
  /// \param xp       The current point.
  /// \param drt      The current moving direction.
  /// \param step_max The upper bound for the step size that makes x feasible.
  /// \param step     In: The initial step length.
  ///                 Out: The calculated step length.
  /// \param fx       In: The objective function value at the current point.
  ///                 Out: The function value at the new point.
  /// \param grad     In: The current gradient vector.
  ///                 Out: The gradient at the new point.
  /// \param dg       In: The inner product between drt and grad.
  ///                 Out: The inner product between drt and the new gradient.
  /// \param x        Out: The new point moved to.
  ///
  template <typename Foo, typename SolverParam>
  static void LineSearch(Foo& f, const SolverParam& param,
                         const Vector& xp, const Vector& drt, const Scalar& step_max,
                         Scalar& step, Scalar& fx, Vector& grad, Scalar& dg, Vector& x)
  {
    using std::abs;
    // std::cout << "========================= Entering line search =========================\n\n";
    
    // Check the value of step
    if (step <= Scalar(0))
      throw std::invalid_argument("'step' must be positive");
    if (step > step_max)
      throw std::invalid_argument("'step' exceeds 'step_max'");
    
    // Save the function value at the current x
    const Scalar fx_init = fx;
    // Projection of gradient on the search direction
    const Scalar dg_init = dg;
    
    // std::cout << "fx_init = " << fx_init << ", dg_init = " << dg_init << std::endl << std::endl;
    
    // Make sure d points to a descent direction
    if (dg_init >= Scalar(0))
      throw std::logic_error("the moving direction does not decrease the objective function value");
    
    // Tolerance for convergence test
    // Sufficient decrease
    const Scalar test_decr = param.ftol * dg_init;
    // Curvature
    const Scalar test_curv = -param.wolfe * dg_init;
    
    // The bracketing interval
    Scalar I_lo = Scalar(0), I_hi = std::numeric_limits<Scalar>::infinity();
    Scalar fI_lo = Scalar(0), fI_hi = std::numeric_limits<Scalar>::infinity();
    Scalar gI_lo = (Scalar(1) - param.ftol) * dg_init, gI_hi = std::numeric_limits<Scalar>::infinity();
    // We also need to save x and grad for step=I_lo, since we want to return the best
    // step size along the path when strong Wolfe condition is not met
    Vector x_lo = xp, grad_lo = grad;
    Scalar fx_lo = fx_init, dg_lo = dg_init;
    
    // Function value and gradient at the current step size
    x.noalias() = xp + step * drt;
    fx = f(x, grad);
    dg = grad.dot(drt);
    
    // std::cout << "max_step = " << step_max << ", step = " << step << ", fx = " << fx << ", dg = " << dg << std::endl;
    
    // Convergence test
    if (fx <= fx_init + step * test_decr && abs(dg) <= test_curv)
    {
      // std::cout << "** Criteria met\n\n";
      // std::cout << "========================= Leaving line search =========================\n\n";
      return;
    }
    
    // Extrapolation factor
    const Scalar delta = Scalar(1.1);
    int iter;
    for (iter = 0; iter < param.max_linesearch; iter++)
    {
      // ft = psi(step) = f(xp + step * drt) - f(xp) - step * test_decr
      // gt = psi'(step) = dg - mu * dg_init
      // mu = param.ftol
      const Scalar ft = fx - fx_init - step * test_decr;
      const Scalar gt = dg - param.ftol * dg_init;
      
      // Update step size and bracketing interval
      Scalar new_step;
      if (ft > fI_lo)
      {
        // Case 1: ft > fl
        new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);
        // Sanity check: if the computed new_step is too small, typically due to
        // extremely large value of ft, switch to the middle point
        if (new_step <= param.min_step)
          new_step = (I_lo + step) / Scalar(2);
        
        I_hi = step;
        fI_hi = ft;
        gI_hi = gt;
        
        // std::cout << "Case 1: new step = " << new_step << std::endl;
      }
      else if (gt * (I_lo - step) > Scalar(0))
      {
        // Case 2: ft <= fl, gt * (al - at) > 0
        //
        // Page 291 of Moré and Thuente (1994) suggests that
        // newat = min(at + delta * (at - al), amax), delta in [1.1, 4]
        new_step = std::min(step_max, step + delta * (step - I_lo));
        
        // We can also consider the following scheme:
        // First let step_selection() decide a value, and then project to the range above
        //
        // new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);
        // const Scalar delta2 = Scalar(4)
        // const Scalar t1 = step + delta * (step - I_lo);
        // const Scalar t2 = step + delta2 * (step - I_lo);
        // const Scalar tl = std::min(t1, t2), tu = std::max(t1, t2);
        // new_step = std::min(tu, std::max(tl, new_step));
        // new_step = std::min(step_max, new_step);
        
        I_lo = step;
        fI_lo = ft;
        gI_lo = gt;
        // Move x and grad to x_lo and grad_lo, respectively
        x_lo.swap(x);
        grad_lo.swap(grad);
        fx_lo = fx;
        dg_lo = dg;
        
        // std::cout << "Case 2: new step = " << new_step << std::endl;
      }
      else
      {
        // Case 3: ft <= fl, gt * (al - at) <= 0
        new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);
        
        I_hi = I_lo;
        fI_hi = fI_lo;
        gI_hi = gI_lo;
        
        I_lo = step;
        fI_lo = ft;
        gI_lo = gt;
        // Move x and grad to x_lo and grad_lo, respectively
        x_lo.swap(x);
        grad_lo.swap(grad);
        fx_lo = fx;
        dg_lo = dg;
        
        // std::cout << "Case 3: new step = " << new_step << std::endl;
      }
      
      // Case 1 and 3 are interpolations, whereas Case 2 is extrapolation
      // This means that Case 2 may return new_step = step_max,
      // and we need to decide whether to accept this value
      // 1. If both step and new_step equal to step_max, it means
      //    step will have no further change, so we accept it
      // 2. Otherwise, we need to test the function value and gradient
      //    on step_max, and decide later
      
      // In case step, new_step, and step_max are equal, directly return the computed x and fx
      if (step == step_max && new_step >= step_max)
      {
        // std::cout << "** Maximum step size reached\n\n";
        // std::cout << "========================= Leaving line search =========================\n\n";
        
        // Move {x, grad}_lo back before returning
        x.swap(x_lo);
        grad.swap(grad_lo);
        return;
      }
      // Otherwise, recompute x and fx based on new_step
      step = new_step;
      
      if (step < param.min_step)
        throw std::runtime_error("the line search step became smaller than the minimum value allowed");
      
      if (step > param.max_step)
        throw std::runtime_error("the line search step became larger than the maximum value allowed");
      
      // Update parameter, function value, and gradient
      x.noalias() = xp + step * drt;
      fx = f(x, grad);
      dg = grad.dot(drt);
      
      // std::cout << "step = " << step << ", fx = " << fx << ", dg = " << dg << std::endl;
      
      // Convergence test
      if (fx <= fx_init + step * test_decr && abs(dg) <= test_curv)
      {
        // std::cout << "** Criteria met\n\n";
        // std::cout << "========================= Leaving line search =========================\n\n";
        return;
      }
      
      // Now assume step = step_max, and we need to decide whether to
      // exit the line search (see the comments above regarding step_max)
      // If we reach here, it means this step size does not pass the convergence
      // test, so either the sufficient decrease condition or the curvature
      // condition is not met yet
      //
      // Typically the curvature condition is harder to meet, and it is
      // possible that no step size in [0, step_max] satisfies the condition
      //
      // But we need to make sure that its psi function value is smaller than
      // the best one so far. If not, go to the next iteration and find a better one
      if (step >= step_max)
      {
        const Scalar ft_bound = fx - fx_init - step * test_decr;
        if (ft_bound <= fI_lo)
        {
          // std::cout << "** Maximum step size reached\n\n";
          // std::cout << "========================= Leaving line search =========================\n\n";
          return;
        }
      }
    }
    
    // If we have used up all line search iterations, then the strong Wolfe condition
    // is not met. We choose not to raise an exception (unless no step satisfying
    // sufficient decrease is found), but to return the best step size so far
    if (iter >= param.max_linesearch)
    {
      // throw std::runtime_error("the line search routine reached the maximum number of iterations");
      
      // First test whether the last step is better than I_lo
      // If yes, return the last step
      const Scalar ft = fx - fx_init - step * test_decr;
      if (ft <= fI_lo)
        return;
      
      // If not, then the best step size so far is I_lo, but it needs to be positive
      if (I_lo <= Scalar(0))
        throw std::runtime_error("the line search routine is unable to sufficiently decrease the function value");
      
      // Return everything with _lo
      step = I_lo;
      fx = fx_lo;
      dg = dg_lo;
      // Move {x, grad}_lo back
      x.swap(x_lo);
      grad.swap(grad_lo);
      return;
    }
  }
};

}  // namespace LBFGSpp




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

#endif  // LBFGSPP_LBFGSB_H
