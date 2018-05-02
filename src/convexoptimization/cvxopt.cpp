#include <complex>
#include <iostream>
#include <functional>
#include <vector>
#include "convexoptimization/cvxopt.hpp"
#include "mkl.h"

namespace ase
{
  double backtracking_line_search( const std::function< double ( std::vector< double > dx ) >& f_obj,
                                   const std::vector< double > grad_f_obj_at_x, std::vector< double > dx,
                                   const double f_obj0, const double alpha, const double beta, const int max_iter )
  {
    // Pre-allocations/-calculations.
    double step_size = 1.0, dx0 = dx[ 0 ], f_obj_new = f_obj( dx );
    int iter = 0, n = dx.size( );
    double grad_f_obj_dx = cblas_ddot( n, grad_f_obj_at_x.data( ), 1, dx.data( ), 1 );

    while( f_obj_new > ( f_obj0+step_size*alpha*grad_f_obj_dx ) && iter < max_iter )
    {
      std::cout << "In loop" << std::endl;
      // Evaluate the objective function using the current step-size.
      cblas_dscal( n, beta, dx.data( ), 1 );
      f_obj_new = f_obj( dx );

      // Increment the counter.
      iter += 1;
    }
    return dx[ 0 ]/dx0;
  }

  double backtracking_line_search( const std::function< double ( std::vector< std::complex< double > > dx ) >& f_obj,
                                   const std::vector< std::complex< double > > grad_f_obj_at_x, std::vector< std::complex< double > > dx,
                                   const double f_obj0, const double alpha, const double beta, const int max_iter )
  {
    return 0.0;
  }
}
