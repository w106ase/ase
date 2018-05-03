/*! \file cvxopt.cpp
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

#include <complex>
#include <functional>
#include <vector>
#include "convexoptimization/cvxopt.hpp"
#include "mkl.h"

namespace ase
{
  double backtracking_line_search( const std::function< double ( std::vector< double > x ) >& f_obj,
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

  double backtracking_line_search( const std::function< double ( std::vector< std::complex< double > > x ) >& f_obj,
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
}
