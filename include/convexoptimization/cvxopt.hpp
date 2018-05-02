#ifndef CVXOPT_H
#define CVXOPT_H
#include <complex>
#include <functional>
#include <vector>

/*! \file cvxopt.hpp
    \brief Convex optimization functionality using Intel MKL functionality.

    Details.
*/

namespace ase
{
  double backtracking_line_search( const std::function< double ( std::vector< double > dx ) >& f_obj,
                                   const std::vector< double > grad_f_obj_at_x, std::vector< double > dx,
                                   const double f_obj0, const double alpha = 0.45, const double beta = 0.8, const int max_iter = 100 );

  double backtracking_line_search( const std::function< double ( std::vector< std::complex< double > > dx ) >& f_obj,
                                   const std::vector< std::complex< double > > grad_f_obj_at_x, std::vector< std::complex< double > > dx,
                                   const double f_obj0, const double alpha = 0.45, const double beta = 0.8, const int max_iter = 100 );
}
#endif // CVXOPT_H
