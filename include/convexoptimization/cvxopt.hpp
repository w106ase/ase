/*! \file cvxopt.hpp
    \brief Convex optimization functionality implemented using Intel MKL
    routines.

    Convex optimization functionality using Intel MKL functionality. This
    functionality includes line search routines, gradient descent routines,
    steepest descent routines, Newton's method routines, and various generic and
    specialized solvers for convex optimization problems. Much of the provided
    functionality is project driven with the application being radar signal
    processing. Thus, where appropriate routines exist for handling both real-
    and complex-valued inputs.
*/

#ifndef CVXOPT_H
#define CVXOPT_H
#include <complex>
#include <functional>
#include <vector>

namespace ase
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
      \note the descent direction, <tt>dx</tt>, is overwritten by <tt>step_size*dx</tt> in the function call, and
      <tt>x</tt> is overwriten by <tt>x+step_size*dx</tt>.
  */
  double backtracking_line_search( const std::function< double ( const std::vector< double >& x ) >& f_obj,
                                   std::vector< double >& x, const std::vector< double >& grad_f_obj_at_x, std::vector< double >& dx,
                                   const double& alpha = 0.45, const double& beta = 0.8, const int& max_iter = 100 );

  /*! \brief Backtracking line search (see \cite Boyd2004_ase Pg. 464) for determining a step-size, which minimizes a provided objective function that takes complex-valued arguments.

      \param f_obj objective function which takes a single, complex-valued argument <tt>x</tt>, and evaluates <tt>f_obj(x)</tt>.
      \param x current location to take a step from.
      \param grad_f_obj_at_x gradient of the objective function at the current location <tt>x</tt>.
      \param dx descent direction to determine a step-size for.
      \param alpha backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param beta backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param max_iter maximum number of iterations to apply the backtracking line search.

      \return step-size for scaling the descent direction.
      \note the descent direction, <tt>dx</tt>, is overwritten by <tt>step_size*dx</tt> in the function call, and
      <tt>x</tt> is overwriten by <tt>x+step_size*dx</tt>.
  */
  double backtracking_line_search( const std::function< double ( const std::vector< std::complex< double > >& x ) >& f_obj,
                                   std::vector< std::complex< double > >& x, const std::vector< std::complex< double > >& grad_f_obj_at_x,
                                   std::vector< std::complex< double > >& dx, const double& alpha = 0.45, const double& beta = 0.8,
                                   const int& max_iter = 100 );

  /*! \brief General descent method with backtracking line search (see \cite Boyd2004_ase Pg. 464) for an objective function taking a real-valued argument.

      \param f_obj objective function which takes a single, real-valued argument <tt>x</tt>, and evaluates <tt>f_obj(x)</tt>.
      \param grad_f_obj gradient function for the objective function at the location <tt>x</tt>.
      \param desc_dir descent direction at the location <tt>x</tt>.
      \param x starting point to begin the general descent method.
      \param is_grad_desc flag which is true if gradient descent is to be performed.
      \param max_iter maximum number of iterations to apply the general descent method.
      \param norm2_grad_thresh 2-norm threshold of the gradient for stopping the general descent method.
      \param alpha backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param beta backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param max_btls_iter maximum number of iterations to apply the backtracking line search.

      \note the initial point, <tt>x</tt>, is overwritten by <tt>x+step_size*dx</tt> in the function call.
  */
  void general_descent_method_with_btls( const std::function< double ( const std::vector< double >& x ) >& f_obj,
                                         const std::function< void ( const std::vector< double >& x, std::vector< double >& grad_f_obj_at_x ) >& grad_f_obj,
                                         const std::function< void ( const std::vector< double >& x, std::vector< double >& dx ) >& desc_dir,
                                         std::vector< double >& x, const bool& is_grad_desc = false, const int& max_iter = 100,
                                         const double& norm2_grad_thresh = pow( 10.0, -6.0 ), const double& alpha = 0.45,
                                         const double& beta = 0.8, const int& max_btls_iter = 100 );

  /*! \brief General descent method with backtracking line search (see \cite Boyd2004_ase Pg. 464) for an objective function taking a complex-valued argument.

      \param f_obj objective function which takes a single, complex-valued argument <tt>x</tt>, and evaluates <tt>f_obj(x)</tt>.
      \param grad_f_obj gradient function for the objective function at the location <tt>x</tt>.
      \param desc_dir descent direction at the location <tt>x</tt>.
      \param x starting point to begin the general descent method.
      \param is_grad_desc flag which is true if gradient descent is to be performed.
      \param max_iter maximum number of iterations to apply the general descent method.
      \param norm2_grad_thresh 2-norm threshold of the gradient for stopping the general descent method.
      \param alpha backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param beta backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param max_btls_iter maximum number of iterations to apply the backtracking line search.

      \note the initial point, <tt>x</tt>, is overwritten by <tt>x+step_size*dx</tt> in the function call.
  */
  void general_descent_method_with_btls( const std::function< double ( const std::vector< std::complex< double > >& x ) >& f_obj,
                                         const std::function< void ( const std::vector< std::complex< double > >& x, std::vector< std::complex< double > >& grad_f_obj_at_x ) >& grad_f_obj,
                                         const std::function< void ( const std::vector< std::complex< double > >& x, std::vector< std::complex< double > >& dx ) >& desc_dir,
                                         std::vector< std::complex< double > >& x, const bool& is_grad_desc = false,
                                         const int& max_iter = 100, const double& norm2_grad_thresh = pow( 10.0, -6.0 ),
                                         const double& alpha = 0.45, const double& beta = 0.8, const int& max_btls_iter = 100 );
}
#endif // CVXOPT_H
