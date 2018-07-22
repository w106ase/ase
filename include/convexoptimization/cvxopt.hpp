/*! \file cvxopt.hpp
    \brief Convex optimization functionality implemented using Intel MKL
    routines.

    Convex optimization functionality implemented using Intel MKL functionality.
    This functionality includes line search routines, general descent methods,
    general barrier methods, and various generic and specialized solvers for
    convex optimization problems. Much of the provided functionality is project
    driven with the application being radar signal processing. Thus, where
    appropriate routines exist for handling both real- and complex-valued
    inputs.
*/

#ifndef CVXOPT_H
#define CVXOPT_H
#include <complex>
#include <functional>
#include <vector>
#include "constants/constants.hpp"

namespace ase
{
namespace cvx
{
  /*! \brief Backtracking line search (see \cite Boyd2004_ase Pg. 464) for determining a step-size, which minimizes a provided objective function that takes real-valued arguments.

      \param f_obj objective function which takes a single, real-valued argument <tt>x</tt>, and evaluates <tt>f_obj(x)</tt>.
      \param x current location to take a step from.
      \param grad_f_obj_at_x gradient of the objective function at the current location <tt>x</tt>.
      \param dx descent direction to determine a step-size for.
      \param alpha backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param beta backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param max_iter maximum number of iterations to apply the backtracking line search.

      \return step-size for scaling the descent direction.
      \note The descent direction, <tt>dx</tt>, is overwritten by <tt>step_size*dx</tt> in the function call, and
      <tt>x</tt> is overwriten by <tt>x+step_size*dx</tt>.
      \note If the maximum number of iterations is reached, the step-size is set to zero to prevent stepping to an infeasible point.
  */
  double backtracking_line_search( const std::function< double ( const std::vector< double >& x ) >& f_obj,
                                   std::vector< double >& x, const std::vector< double >& grad_f_obj_at_x, std::vector< double >& dx,
                                   const double& alpha = ase::constants::btls_alpha, const double& beta = ase::constants::btls_beta,
                                   const int& max_iter = ase::constants::btls_max_iter );

  /*! \brief Backtracking line search (see \cite Boyd2004_ase Pg. 464) for determining a step-size, which minimizes a provided objective function that takes complex-valued arguments.

      \param f_obj objective function which takes a single, complex-valued argument <tt>x</tt>, and evaluates <tt>f_obj(x)</tt>.
      \param x current location to take a step from.
      \param grad_f_obj_at_x gradient of the objective function at the current location <tt>x</tt>.
      \param dx descent direction to determine a step-size for.
      \param alpha backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param beta backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param max_iter maximum number of iterations to apply the backtracking line search.

      \return step-size for scaling the descent direction.
      \note The descent direction, <tt>dx</tt>, is overwritten by <tt>step_size*dx</tt> in the function call, and
      <tt>x</tt> is overwriten by <tt>x+step_size*dx</tt>.
      \note If the maximum number of iterations is reached, the step-size is set to zero to prevent stepping to an infeasible point.
  */
  double backtracking_line_search( const std::function< double ( const std::vector< std::complex< double > >& x ) >& f_obj,
                                   std::vector< std::complex< double > >& x, const std::vector< std::complex< double > >& grad_f_obj_at_x,
                                   std::vector< std::complex< double > >& dx, const double& alpha = ase::constants::btls_alpha,
                                   const double& beta = ase::constants::btls_beta, const int& max_iter = ase::constants::btls_max_iter );

  /*! \brief General descent method with backtracking line search (see \cite Boyd2004_ase Pg. 464) for an objective function taking a real-valued argument (i.e., \f$ \min_{x \in R^{n}} f(x) \f$).

      \param f_obj objective function which takes a single, real-valued argument <tt>x</tt>, and evaluates <tt>f_obj(x)</tt>.
      \param grad_f_obj gradient function for the objective function at the location <tt>x</tt>.
      \param desc_dir descent direction at the location <tt>x</tt>.
      \param x starting point to begin the general descent method.
      \param max_iter maximum number of iterations to apply the general descent method.
      \param norm2_grad_thresh 2-norm threshold of the gradient for stopping the general descent method.
      \param rel_dx_norm2_thresh 2-norm relative change in <tt>x</tt> threshold to continue processing
      \param alpha backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param beta backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param max_btls_iter maximum number of iterations to apply the backtracking line search.

      \return flag indicating the general descent method terminated early as a
      result of a backtracking line search step-size of zero. This may indicate
      different backtracking line search parameters are needed.

      \note The initial point, <tt>x</tt>, is overwritten by <tt>x+step_size*dx</tt> in the function call.
  */
  bool general_descent_method_with_btls( const std::function< double ( const std::vector< double >& x ) >& f_obj,
                                         const std::function< void ( const std::vector< double >& x, std::vector< double >& grad_f_obj_at_x ) >& grad_f_obj,
                                         const std::function< void ( const std::vector< double >& x, const std::vector< double >& grad_f_obj_at_x, std::vector< double >& dx ) >& desc_dir,
                                         std::vector< double >& x, const int& max_iter = ase::constants::gdm_max_iter, const double& norm2_grad_thresh = ase::constants::gdm_norm2_grad_thresh,
                                         const double& rel_dx_norm2_thresh = ase::constants::gdm_rel_dx_norm2_thresh,
                                         const double& alpha = ase::constants::btls_alpha, const double& beta = ase::constants::btls_beta, const int& max_btls_iter = ase::constants::btls_max_iter );

  /*! \brief General descent method with backtracking line search (see \cite Boyd2004_ase Pg. 464) for an objective function taking a complex-valued argument (i.e., \f$ \min_{x \in C^{n}} f(x) \f$).

      \param f_obj objective function which takes a single, complex-valued argument <tt>x</tt>, and evaluates <tt>f_obj(x)</tt>.
      \param grad_f_obj gradient function for the objective function at the location <tt>x</tt>.
      \param desc_dir descent direction at the location <tt>x</tt>.
      \param x starting point to begin the general descent method.
      \param max_iter maximum number of iterations to apply the general descent method.
      \param norm2_grad_thresh 2-norm threshold of the gradient for stopping the general descent method.
      \param rel_dx_norm2_thresh 2-norm relative change in <tt>x</tt> threshold to continue processing
      \param alpha backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param beta backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param max_btls_iter maximum number of iterations to apply the backtracking line search.

      \return flag indicating the general descent method terminated early as a
      result of a backtracking line search step-size of zero. This may indicate
      different backtracking line search parameters are needed.

      \note The initial point, <tt>x</tt>, is overwritten by <tt>x+step_size*dx</tt> in the function call.
  */
  bool general_descent_method_with_btls( const std::function< double ( const std::vector< std::complex< double > >& x ) >& f_obj,
                                         const std::function< void ( const std::vector< std::complex< double > >& x, std::vector< std::complex< double > >& grad_f_obj_at_x ) >& grad_f_obj,
                                         const std::function< void ( const std::vector< std::complex< double > >& x, const std::vector< std::complex< double > >& grad_f_obj_at_x, std::vector< std::complex< double > >& dx ) >& desc_dir,
                                         std::vector< std::complex< double > >& x, const int& max_iter = ase::constants::gdm_max_iter, const double& norm2_grad_thresh = ase::constants::gdm_norm2_grad_thresh,
                                         const double& rel_dx_norm2_thresh = ase::constants::gdm_rel_dx_norm2_thresh,
                                         const double& alpha = ase::constants::btls_alpha, const double& beta = ase::constants::btls_beta, const int& max_btls_iter = ase::constants::btls_max_iter );

  /*! \brief General barrier method (see \cite Boyd2004_ase Pg. 569) for an objective function taking a real-valued argument.

      \param f_center function for carrying out the centering step (see \cite Boyd2004_ase Pg. 569).
      \param x starting point to begin the barrier method from.
      \param barrier_parameter0 weighting for the original objective function (see \f$t\f$ on Pg. 569 of \cite Boyd2004_ase).
      \param barrier_parameter_update update factor for the barrier parameter (see \f$\mu\f$ on Pg. 569 of \cite Boyd2004_ase).
      \param max_iter maximum number of iterations to apply the general barrier method.
      \param sub_optimality_thresh sub-optimality threshold for stopping the general barrier method.

      \return int indicating the number of outer iterations completed.

      \note The initial point, <tt>x</tt>, is overwritten in the function call.
  */
  int general_barrier_method( const std::function< bool ( std::vector< double >& x, const double& barrier_parameter ) >& f_center,
                               std::vector< double >& x, const double& barrier_parameter0 = ase::constants::gbm_barrier_param0,
                               const double& barrier_parameter_update = ase::constants::gbm_barrier_param_update,
                               const int& max_iter = ase::constants::gbm_max_iter, const double& sub_optimality_thresh = ase::constants::gbm_sub_opt_thresh );

  /*! \brief Logarithmic barrier function \f$-\sum_{i=0}^{n-1} \mathrm{ln}\left( -[x]_{i}\right)\f$ (see \cite Boyd2004_ase Pg. 563).

      \param x the input vector to calculate the logarithmic barrier function for.
      \param negate_x flag which is true when computing \f$-\sum_{i=0}^{n-1} \mathrm{ln}\left( -[-x]_{i} )\right)\f$

      \return logarithmic barrier function value (i.e., \f$-\sum_{i=0}^{n-1} \mathrm{ln}\left(\left[ x\right]_{i}\right)\f$ or \f$-\sum_{i=0}^{n-1} \mathrm{ln}\left(-\left[ x\right]_{i}\right)\f$).

      \note The flag <tt>negate_x</tt> should be set so that \f$-x \succ 0\f$.
  */
  double log_barrier( const std::vector< double >& x, const bool& negate_x = false );

  void dual_sdp_inequality_form_with_diag_plus_low_rank_lmi( const std::vector< double >& c,
                                                             std::vector< double >& Z,
                                                             std::vector< double >& lmi_inv,
                                                             const double& precision = ase::constants::med_prec );

  /*! \brief Inequality form SDP (see \cite Boyd2004_ase Pg. 169) interior point method, with a diagonal plus low-rank data linear matrix inequality (LMI)
      \f{eqnarray*}{
        \min_{x \in R^{n}} &c^{\mathrm{T}} x& \\
        \textrm{subject to} &\mathrm{Diag}(x)+ZZ^{\mathrm{T}} \preceq 0,&
      \f}
      where \f$ c \in R^{n} \f$ and \f$ Z \in R^{n \times p} \f$ are input data.

      \param x the optimization variable.
      \param c input data vector defining the objective function.
      \param Z input data matrix defining the inequality constraint.
      \param lmi_inv linear matrix inequality (LMI) inverse (i.e., \f$\left( \mathrm{Diag}(x)+ZZ^{\mathrm{T}} \right)^{-1} \f$)
      \param precision desired precision for the solution of the problem

      \note The input data matrix <tt>Z</tt> does not have a const because the
      data is scaled. However, before a solution is returned the data matrix is
      scaled back to its original value. Thus, even though there is no const
      with <tt>Z</tt> the value is not changed.
  */
  void sdp_inequality_form_with_diag_plus_low_rank_lmi( std::vector< double >& x,
                                                        const std::vector< double >& c,
                                                        std::vector< double >& Z,
                                                        std::vector< double >& lmi_inv,
                                                        const double& precision = ase::constants::med_prec );

  /*! \brief Inequality form SDP (see \cite Boyd2004_ase Pg. 169) interior point method, with a diagonal plus low-rank data linear matrix inequality (LMI)
      \f{eqnarray*}{
        \min_{x \in R^{n}} &c^{\mathrm{T}} x& \\
        \textrm{subject to} &\mathrm{Diag}(x)+ZZ^{\mathrm{H}} \preceq 0,&
      \f}
      where \f$ c \in R^{n} \f$ and \f$ Z \in C^{n \times p} \f$ are input data.

      \param x the optimization variable.
      \param c input data vector defining the objective function.
      \param Z input data matrix defining the inequality constraint.
      \param lmi_inv linear matrix inequality (LMI) inverse (i.e., \f$\left( \mathrm{Diag}(x)+ZZ^{\mathrm{T}} \right)^{-1} \f$)
      \param precision desired precision for the solution of the problem

      \note The input data matrix <tt>Z</tt> does not have a const because the
      data is scaled. However, before a solution is returned the data matrix is
      scaled back to its original value. Thus, even though there is no const
      with <tt>Z</tt> the value is not changed.
  */
  void sdp_inequality_form_with_diag_plus_low_rank_lmi( std::vector< double >& x,
                                                        const std::vector< double >& c,
                                                        std::vector< std::complex< double > >& Z,
                                                        std::vector< std::complex< double > >& lmi_inv,
                                                        const double& precision = ase::constants::med_prec );

  // /*! \brief Inequality form SDP (see \cite Boyd2004_ase Pg. 169) interior point method, with a diagonal plus data linear matrix inequality (LMI)
  // \f{eqnarray*}{
  //   \min_{x \in R^{n}} &c^{\mathrm{T}} x& \\
  //   \textrm{subject to} &\mathrm{Diag}(x)+Z \preceq 0,&
  // \f}
  // where \f$ c \in R^{n} \f$ and \f$ Z \in R^{n \times n} \f$ are input data.
  //
  // */
  // void sdp_inequality_form_with_diag_plus_data_lmi( );
  //
  // /*! \brief Inequality form SDP (see \cite Boyd2004_ase Pg. 169) interior point method, with a diagonal plus data linear matrix inequality (LMI)
  // \f{eqnarray*}{
  //   \min_{x \in R^{n}} &c^{\mathrm{T}} x& \\
  //   \textrm{subject to} &\mathrm{Diag}(x)+Z \preceq 0,&
  // \f}
  // where \f$ c \in R^{n} \f$ and \f$ Z \in C^{n \times n} \f$ are input data.
  //
  // */
  // void sdp_inequality_form_with_diag_plus_data_lmi( );
} // namespace cvx
} // namespace ase
#endif // CVXOPT_H
