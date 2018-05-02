#include <iostream>
#include <complex>
#include <vector>
#include "convexoptimization/cvxopt.hpp"
#include "mkl.h"

/*! \file testBacktrackingLineSearch.cpp
    \brief Demonstrates the backtracking line search function using both real
    and complex-valued data.

    Details.
*/

using namespace std;

int main( int argc, char* argv[ ])
{
  // Dimension of the problem.
  int n = 5;

  /* Random start point. The objective function, which we will attempt to
  minimize using the backtracking line search is x^{T} x. */
  vector< double > x( n ), dx( n );
  double f_obj0 = 0.0, grad_f_obj_dx = 0.0;
  for( int i = 0; i < n; i++ )
  {
    x[ i ] = rand( );
    dx[ i ] = -x[ i ];
    f_obj0 += pow( x[ i ], 2.0 );
    grad_f_obj_dx += x[ i ]*dx[ i ];
    if( i < n-1 )
      cout << "x = ( " << x[ i ] << ", ";
    else
      cout << x[ i ] << " )" << endl;
  }
  cout << "Initial objective function value: " << f_obj0 << endl;

  /* Define a lambda which takes one argument, the step direction scaled by the
  step-size, that defines the objective function x^{T} x. The objective function
  is to be minimized. */
  auto f_obj = [ &x ]( vector< double > dx ) -> double
  {
    double norm2 = 0.0;
    for( int i = 0; i < dx.size( ); i++ )
      norm2 += pow( x[ i ]+dx[ i ], 2.0 );
    return norm2;
  };

  // for( int i = 0; i < 1; i++ )
  // {
    ase::backtracking_line_search( f_obj, x, dx, f_obj0 );
    cblas_daxpy( n, 1.0, dx.data( ), 1, x.data( ), 1 );
    cout << x[ 1 ] << endl;
  // }

  return 0;
}
