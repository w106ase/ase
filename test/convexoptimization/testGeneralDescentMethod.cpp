/*! \file testGeneralDescentMethod.cpp
    \brief Demonstrates the general descent method functions using both real-
    and complex-valued data (i.e., \f$ \min_{x \in R^{n}} f(x) \f$ and \f$ \min_{x \in C^{n}} f(x) \f$).

    Demonstrates the general descent method functions using real- and
    complex-valued data. For the demonstrations, the example problem 9.3.2 on
    Pg. 469 of \cite Boyd2004_ase is used. The specific functions that are
    exercised is/are:
    1. general_descent_method_with_btls()
    2. general_descent_method_with_btls()
*/

#include <iostream>
#include <complex>
#include <vector>
#include "convexoptimization/cvxopt.hpp"
#include "mkl.h"

using namespace std;

#ifndef DOXYGEN_SKIP
void real_valued_example( );
void complex_valued_example( );

int main( int argc, char* argv[ ])
{
  // Real-valued example.
  cout << "Example Result Using Real-Valued Arguments" << endl;
  cout << "-----------------------------------------------------------" << endl;
  cout << endl;
  srand( 0 );
  real_valued_example( );
  cout << endl;

  // Complex-valued example.
  cout << "Example Result Using Complex-Valued Arguments" << endl;
  cout << "-----------------------------------------------------------" << endl;
  cout << endl;
  srand( 0 );
  complex_valued_example( );
  cout << endl;

  return 0;
}

void real_valued_example( )
{
  /* Dimension of the problem, an initial starting point, and the objective
  function. */
  int n = 2;
  double gamma = 2.0;
  vector< double > x = { 10.0, 1.0 };
  auto f_obj = [ &gamma ]( const vector< double >& x ) -> double
  {
    return 0.5*( pow( x[ 0 ], 2.0 )+gamma*pow( x[ 1 ], 2.0 ));
  };

  // Define a lambda for the gradient and descent direction.
  auto grad_f_obj = [ &gamma ]( const vector< double >& x,
                                vector< double >& grad_f_obj_at_x ) -> void
  {
    grad_f_obj_at_x[ 0 ] = x[ 0 ];
    grad_f_obj_at_x[ 1 ] = gamma*x[ 1 ];
  };

  auto desc_dir = [ &n ]( const vector< double >& x,
                          const vector< double >& grad_f_obj_at_x,
                          vector< double >& dx ) -> void
  {
    dx = grad_f_obj_at_x;
    cblas_dscal( n, -1.0, dx.data( ), 1 );
  };

  // Apply the general descent method.
  cout << "Starting point: ( " << x[ 0 ] << ", " << x[ 1 ] << " )" << endl;
  cout << "Initial objective function value: " << f_obj( x ) << endl;
  cout << endl;

  ase::cvx::general_descent_method_with_btls( f_obj, grad_f_obj, desc_dir, x );

  cout << "Ending point: ( " << x[ 0 ] << ", " << x[ 1 ] << " )" << endl;
  cout << "Ending objective function value: " << f_obj( x ) << endl;
}

void complex_valued_example( )
{
  /* Dimension of the problem, an initial starting point, and the objective
  function. */
  int n = 2;
  double gamma = 2.0;
  vector< complex< double > > x = {{ 10.0, 10.0 }, { 1.0, 2.0 }};
  auto f_obj = [ &gamma ]( const vector< complex< double > >& x ) -> double
  {
    return pow( abs( x[ 0 ]), 2.0 )+gamma*pow( abs( x[ 1 ]), 2.0 );
  };

  // Define a lambda for the gradient and descent direction.
  auto grad_f_obj = [ &gamma ]( const vector< complex< double > >& x,
                                vector< complex< double > >& grad_f_obj_at_x ) -> void
  {
    grad_f_obj_at_x[ 0 ] = x[ 0 ];
    grad_f_obj_at_x[ 1 ] = gamma*x[ 1 ];
  };

  auto desc_dir = [ &n ]( const vector< complex< double > >& x,
                          const vector< complex< double > >& grad_f_obj_at_x,
                          vector< complex< double > >& dx ) -> void
  {
    dx = grad_f_obj_at_x;
    cblas_zdscal( n, -1.0, dx.data( ), 1 );
  };

  // Apply the general descent method.
  cout << "Starting point: ( " << x[ 0 ] << ", " << x[ 1 ] << " )" << endl;
  cout << "Initial objective function value: " << f_obj( x ) << endl;
  cout << endl;

  ase::cvx::general_descent_method_with_btls( f_obj, grad_f_obj, desc_dir, x );

  cout << "Ending point: ( " << x[ 0 ] << ", " << x[ 1 ] << " )" << endl;
  cout << "Ending objective function value: " << f_obj( x ) << endl;
}
#endif
