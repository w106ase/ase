/*! \file cvxopt.hpp
    \brief Convex optimization functionality using Intel MKL functionality.

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
  /*! \fn double backtracking_line_search( const std::function< double ( std::vector< double > dx ) >& f_obj,
                                           const std::vector< double > grad_f_obj_at_x, std::vector< double > dx,
                                           const double f_obj0, const double alpha = 0.45, const double beta = 0.8, const int max_iter = 100 )
      \brief Backtracking line search (see \cite Boyd2004_ase Pg. 464) for determining a step-size, which minimizes a provided objective function that takes real-valued arguments.

      \param f_obj objective function which takes a single, real-valued argument <tt>dx</tt>, and evaluates <tt>f_obj(x+dx)</tt>.
      \param grad_f_obj_at_x gradient of the objective function at the current location <tt>x</tt>
      \param dx descent direction to determine a step-size for (overwritten with <tt>step_size*dx</tt>).
      \param f_obj0 initial objective function value at <tt>x</tt> (i.e., <tt>f_obj(x)</tt>).
      \param alpha backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param beta backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param max_iter maximum number of iterations to apply the backtracking line search.
  */
  double backtracking_line_search( const std::function< double ( std::vector< double > dx ) >& f_obj,
                                   const std::vector< double > grad_f_obj_at_x, std::vector< double > dx,
                                   const double f_obj0, const double alpha = 0.45, const double beta = 0.8, const int max_iter = 100 );

  /*! \fn double backtracking_line_search( const std::function< double ( std::vector< std::complex< double > > dx ) >& f_obj,
                                           const std::vector< std::complex< double > > grad_f_obj_at_x, std::vector< std::complex< double > > dx,
                                           const double f_obj0, const double alpha = 0.45, const double beta = 0.8, const int max_iter = 100 )
      \brief Backtracking line search (see \cite Boyd2004_ase Pg. 464) for determining a step-size, which minimizes a provided objective function that takes complex-valued arguments.

      \param f_obj objective function which takes a single, complex-valued argument <tt>dx</tt>, and evaluates <tt>f_obj(x+dx)</tt>.
      \param grad_f_obj_at_x gradient of the objective function at the current location <tt>x</tt>
      \param dx descent direction to determine a step-size for (overwritten with <tt>step_size*dx</tt>).
      \param f_obj0 initial objective function value at <tt>x</tt> (i.e., <tt>f_obj(x)</tt>).
      \param alpha backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param beta backtracking line search parameter (see \cite Boyd2004_ase Pg. 464).
      \param max_iter maximum number of iterations to apply the backtracking line search.
  */
  double backtracking_line_search( const std::function< double ( std::vector< std::complex< double > > dx ) >& f_obj,
                                   const std::vector< std::complex< double > > grad_f_obj_at_x, std::vector< std::complex< double > > dx,
                                   const double f_obj0, const double alpha = 0.45, const double beta = 0.8, const int max_iter = 100 );
}
#endif // CVXOPT_H
