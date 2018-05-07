/*! \file cvxopt.cpp
    \brief Convex optimization functionality implemented using Intel MKL
    routines.

    Convex optimization functionality using Intel MKL functionality. This
    functionality includes line search routines, descent methods, and various
    generic and specialized solvers for convex optimization problems. Much of
    the provided functionality is project driven with the application being
    radar signal processing. Thus, where appropriate routines exist for handling
    both real- and complex-valued inputs.
*/

#include <complex>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>
#include "convexoptimization/cvxopt.hpp"
#include "mkl.h"

namespace ase
{
namespace cvx
{
  double backtracking_line_search( const std::function< double ( const std::vector< double >& x ) >& f_obj,
                                   std::vector< double >& x, const std::vector< double >& grad_f_obj_at_x, std::vector< double >& dx,
                                   const double& alpha, const double& beta, const int& max_iter )
  {
    // Pre-allocations/-calculations.
    std::vector< double > x0 = x; // Retain copy of the starting point
    double step_size = 1.0, f_obj0 = f_obj( x );
    int iter = 0, n = dx.size( );
    double grad_f_obj_dx = cblas_ddot( n, grad_f_obj_at_x.data( ), 1, dx.data( ), 1 );

    // Update x using a unit step-size and evaluate the objective function.
    cblas_daxpy( n, step_size, dx.data( ), 1, x.data( ), 1 );
    double f_obj_new = f_obj( x );

    while( f_obj_new > ( f_obj0+step_size*alpha*grad_f_obj_dx ) && iter < max_iter )
    {
      // Evaluate the objective function using the current step-size.
      step_size *= beta;
      x = x0;
      cblas_daxpy( n, step_size, dx.data( ), 1, x.data( ), 1 );
      f_obj_new = f_obj( x );

      // Increment the counter.
      iter += 1;
    }
    return step_size;
  }

  double backtracking_line_search( const std::function< double ( const std::vector< std::complex< double > >& x ) >& f_obj,
                                   std::vector< std::complex< double > >& x, const std::vector< std::complex< double > >& grad_f_obj_at_x,
                                   std::vector< std::complex< double > >& dx, const double& alpha, const double& beta, const int& max_iter )
  {
    // Pre-allocations/-calculations.
    std::vector< std::complex< double > > x0 = x; // Retain copy of the starting point
    double step_size = 1.0, f_obj0 = f_obj( x );
    int iter = 0, n = dx.size( );

    /* NOTE: this quantity should always be real-valued. To enforce this only
    the real portion is retained. */
    std::complex< double > grad_f_obj_dx_cplx;
    cblas_zdotc_sub( n, grad_f_obj_at_x.data( ), 1, dx.data( ), 1, &grad_f_obj_dx_cplx );
    double grad_f_obj_dx = real( grad_f_obj_dx_cplx );

    // Update x using a unit step-size and evaluate the objective function.
    cblas_zaxpy( n, &step_size, dx.data( ), 1, x.data( ), 1 );
    double f_obj_new = f_obj( x );

    while( f_obj_new > ( f_obj0+step_size*alpha*grad_f_obj_dx ) && iter < max_iter )
    {
      // Evaluate the objective function using the current step-size.
      step_size *= beta;
      x = x0;
      cblas_zaxpy( n, &step_size, dx.data( ), 1, x.data( ), 1 );
      f_obj_new = f_obj( x );

      // Increment the counter.
      iter += 1;
    }
    return step_size;

  }

  void general_descent_method_with_btls( const std::function< double ( const std::vector< double >& x ) >& f_obj,
                                         const std::function< void ( const std::vector< double >& x, std::vector< double >& grad_f_obj_at_x ) >& grad_f_obj,
                                         const std::function< void ( const std::vector< double >& x, const std::vector< double >& grad_f_obj_at_x, std::vector< double >& dx ) >& desc_dir,
                                         std::vector< double >& x, const int& max_iter, const double& norm2_grad_thresh,
                                         const double& alpha, const double& beta, const int& max_btls_iter )
  {
    // Pre-allocations/-calculations.
    int iter = 0, n = x.size( );
    std::vector< double > grad_f_obj_at_x( n ), dx( n );

    while( iter < max_iter )
    {
      // Compute the gradient at the current point and a descent direction.
      grad_f_obj( x, grad_f_obj_at_x );
      desc_dir( x, grad_f_obj_at_x, dx );

      // Check the stopping criterion.
      if( cblas_dnrm2( n, grad_f_obj_at_x.data( ), 1 ) < norm2_grad_thresh )
        break;

      /* Backtracking line search.
      NOTE: x is overwritten in the function call with x+step_size*dx. */
      backtracking_line_search( f_obj, x, grad_f_obj_at_x, dx, alpha, beta, max_btls_iter );

      // Increment the counter.
      iter += 1;
    }
  }

  void general_descent_method_with_btls( const std::function< double ( const std::vector< std::complex< double > >& x ) >& f_obj,
                                         const std::function< void ( const std::vector< std::complex< double > >& x, std::vector< std::complex< double > >& grad_f_obj_at_x ) >& grad_f_obj,
                                         const std::function< void ( const std::vector< std::complex< double > >& x, const std::vector< std::complex< double > >& grad_f_obj_at_x, std::vector< std::complex< double > >& dx ) >& desc_dir,
                                         std::vector< std::complex< double > >& x, const int& max_iter, const double& norm2_grad_thresh, const double& alpha, const double& beta, const int& max_btls_iter )
  {
    // Pre-allocations/-calculations.
    int iter = 0, n = x.size( );
    std::vector< std::complex< double > > grad_f_obj_at_x( n ), dx( n );

    while( iter < max_iter )
    {
      // Compute the gradient at the current point and a descent direction.
      grad_f_obj( x, grad_f_obj_at_x );
      desc_dir( x, grad_f_obj_at_x, dx );

      // Check the stopping criterion.
      if( cblas_dznrm2( n, grad_f_obj_at_x.data( ), 1 ) < norm2_grad_thresh )
        break;

      /* Backtracking line search.
      NOTE: x is overwritten in the function call with x+step_size*dx. */
      backtracking_line_search( f_obj, x, grad_f_obj_at_x, dx, alpha, beta, max_btls_iter );

      // Increment the counter.
      iter += 1;
    }
  }

  void sdp_inequality_form_with_diag_plus_low_rank_lmi( )
  {

  }

  void sdp_inequality_form_with_diag_plus_data_lmi( )
  {

  }
} // namespace cvx
} // namespace ase
