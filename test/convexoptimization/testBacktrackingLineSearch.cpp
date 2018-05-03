/*! \file testBacktrackingLineSearch.cpp
    \brief Demonstrates the backtracking line search functions using both real-
    and complex-valued data.

    Demonstrates the backtracking line search functions using real- and
    complex-valued data. For the demonstrations, the example problem 9.3.2 on
    Pg. 469 of \cite Boyd2004_ase is used. The specific functions that are
    exercised is/are:
    1. backtracking_line_search()
    2. backtracking_line_search()
    3. backtracking_line_search()
    4. backtracking_line_search()
*/

#include <iostream>
#include <complex>
#include <vector>
#include "convexoptimization/cvxopt.hpp"
#include "mkl.h"

using namespace std;

#ifndef DOXYGEN_SKIP
void real_valued_example1( );
void real_valued_example2( );
void complex_valued_example1( );
void complex_valued_example2( );

int main( int argc, char* argv[ ])
{
  // Real-valued examples.
  cout << "Example Result #1 Using Real-Valued Arguments" << endl;
  cout << "-----------------------------------------------------------" << endl;
  cout << endl;
  srand( 0 );
  real_valued_example1( );
  cout << endl;

  cout << "Example Result #2 Using Real-Valued Arguments" << endl;
  cout << "-----------------------------------------------------------" << endl;
  cout << endl;
  srand( 0 );
  real_valued_example2( );
  cout << endl;

  // Complex-valued examples.
  cout << "Example Result #1 Using Complex-Valued Arguments" << endl;
  cout << "-----------------------------------------------------------" << endl;
  cout << endl;
  srand( 0 );
  complex_valued_example1( );
  cout << endl;

  cout << "Example Result #2 Using Complex-Valued Arguments" << endl;
  cout << "-----------------------------------------------------------" << endl;
  cout << endl;
  srand( 0 );
  complex_valued_example2( );
  cout << endl;
  return 0;
}

void real_valued_example1( )
{
  /* Dimension of the problem, an initial starting point, and the objective
  function. */
  int n = 2;
  double gamma = 2.0;
  vector< double > x = { 10.0, 1.0 };
  auto f_obj = [ &gamma ]( vector< double > x ) -> double
  {
    return 0.5*( pow( x[ 0 ], 2.0 )+gamma*pow( x[ 1 ], 2.0 ));
  };

  // Define a lambda for the gradient and descent direction.
  auto grad_f_obj = [ &x, &gamma, &n ]( vector< double >& grad_f_obj_at_x,
                                        vector< double >& dx ) -> void
  {
    grad_f_obj_at_x[ 0 ] = x[ 0 ];
    grad_f_obj_at_x[ 1 ] = gamma*x[ 1 ];
    dx = grad_f_obj_at_x;
    cblas_dscal( n, -1.0, dx.data( ), 1 );
  };

  // Pre-allocations/-calculations.
  vector< double > grad_f_obj_at_x( n ), dx( n );
  double f_obj0 = f_obj( x );

  // Exercise the backtracking line search functionality.
  cout << "Starting point: ( " << x[ 0 ] << ", " << x[ 1 ] << " )" << endl;
  cout << endl;
  cout << "Iteration 0 objective function value: " << f_obj0 << endl;
  for( int i = 0; i < 5; i++ )
  {
    // Assign the gradient and descent direction.
    grad_f_obj( grad_f_obj_at_x, dx );

    /* Backtracking line search followed by taking updating the objective
    function value and taking a step.
    NOTE: dx is overwritten in the backtracking line search function call with
    step_size*dx, and x is overwritten with x+step_size*dx. */
    double step_size = ase::backtracking_line_search( f_obj, x, grad_f_obj_at_x, dx );
    f_obj0 = f_obj( x );

    // Evaluate the new objective function value.
    cout << "Iteration " << i+1 << " objective function value: " << f_obj0 << endl;
  }

  cout << endl;
  cout << "Terminating point: ( " << x[ 0 ] << ", " << x[ 1 ] << " )" << endl;
}

void real_valued_example2( )
{
  /* Dimension of the problem, an initial starting point, and the objective
  function. */
  int n = 2;
  double gamma = 2.0;
  vector< double > x = { 10.0, 1.0 };
  auto f_obj = [ &x, &gamma ]( vector< double > dx ) -> double
  {
    return 0.5*( pow( x[ 0 ]+dx[ 0 ], 2.0 )+gamma*pow( x[ 1 ]+dx[ 1 ], 2.0 ));
  };

  // Define a lambda for the gradient and descent direction.
  auto grad_f_obj = [ &x, &gamma, &n ]( vector< double >& grad_f_obj_at_x,
                                        vector< double >& dx ) -> void
  {
    grad_f_obj_at_x[ 0 ] = x[ 0 ];
    grad_f_obj_at_x[ 1 ] = gamma*x[ 1 ];
    dx = grad_f_obj_at_x;
    cblas_dscal( n, -1.0, dx.data( ), 1 );
  };

  // Pre-allocations/-calculations.
  vector< double > grad_f_obj_at_x( n ), dx( n );
  double f_obj0 = f_obj( dx );

  // Exercise the backtracking line search functionality.
  cout << "Starting point: ( " << x[ 0 ] << ", " << x[ 1 ] << " )" << endl;
  cout << endl;
  cout << "Iteration 0 objective function value: " << f_obj0 << endl;
  for( int i = 0; i < 5; i++ )
  {
    // Assign the gradient and descent direction.
    grad_f_obj( grad_f_obj_at_x, dx );

    /* Backtracking line search followed by taking updating the objective
    function value and taking a step.
    NOTE: dx is overwritten in the backtracking line search function call with
    step_size*dx. */
    double step_size = ase::backtracking_line_search( f_obj, grad_f_obj_at_x, dx, f_obj0 );
    f_obj0 = f_obj( dx );
    cblas_daxpy( n, 1.0, dx.data( ), 1, x.data( ), 1 );

    // Evaluate the new objective function value.
    cout << "Iteration " << i+1 << " objective function value: " << f_obj0 << endl;
  }

  cout << endl;
  cout << "Terminating point: ( " << x[ 0 ] << ", " << x[ 1 ] << " )" << endl;
}

void complex_valued_example1( )
{
  /* Dimension of the problem, an initial starting point, and the objective
  function. */
  int n = 2;
  double gamma = 2.0;
  vector< complex< double > > x = {{ 10.0, 10.0 }, { 1.0, 2.0 }};
  auto f_obj = [ &gamma ]( vector< complex< double > > x ) -> double
  {
    return pow( abs( x[ 0 ]), 2.0 )+gamma*pow( abs( x[ 1 ]), 2.0 );
  };

  // Define a lambda for the gradient and descent direction.
  auto grad_f_obj = [ &x, &gamma, &n ]( vector< complex< double > >& grad_f_obj_at_x,
                                        vector< complex< double > >& dx ) -> void
  {
    grad_f_obj_at_x[ 0 ] = x[ 0 ];
    grad_f_obj_at_x[ 1 ] = gamma*x[ 1 ];
    dx = grad_f_obj_at_x;
    cblas_zdscal( n, -1.0, dx.data( ), 1 );
  };

  // Pre-allocations/-calculations.
  vector< complex< double > > grad_f_obj_at_x( n ), dx( n );
  double f_obj0 = f_obj( x );

  // Exercise the backtracking line search functionality.
  cout << "Starting point: ( " << x[ 0 ] << ", " << x[ 1 ] << " )" << endl;
  cout << endl;
  cout << "Iteration 0 objective function value: " << f_obj0 << endl;
  for( int i = 0; i < 5; i++ )
  {
    // Assign the gradient and descent direction.
    grad_f_obj( grad_f_obj_at_x, dx );

    /* Backtracking line search followed by taking updating the objective
    function value and taking a step.
    NOTE: dx is overwritten in the backtracking line search function call with
    step_size*dx, and x is overwritten with x+step_size*dx. */
    double step_size = ase::backtracking_line_search( f_obj, x, grad_f_obj_at_x, dx );
    f_obj0 = f_obj( x );

    // Evaluate the new objective function value.
    cout << "Iteration " << i+1 << " objective function value: " << f_obj0 << endl;
  }

  cout << endl;
  cout << "Terminating point: ( " << x[ 0 ] << ", " << x[ 1 ] << " )" << endl;
}

void complex_valued_example2( )
{
  /* Dimension of the problem, an initial starting point, and the objective
  function. */
  int n = 2;
  double gamma = 2.0;
  vector< complex< double > > x = {{ 10.0, 10.0 }, { 1.0, 2.0 }};
  auto f_obj = [ &x, &gamma ]( vector< complex< double > > dx ) -> double
  {
    return pow( abs( x[ 0 ]+dx[ 0 ]), 2.0 )+gamma*pow( abs( x[ 1 ]+dx[ 1 ]), 2.0 );
  };

  // Define a lambda for the gradient and descent direction.
  auto grad_f_obj = [ &x, &gamma, &n ]( vector< complex< double > >& grad_f_obj_at_x,
                                        vector< complex< double > >& dx ) -> void
  {
    grad_f_obj_at_x[ 0 ] = x[ 0 ];
    grad_f_obj_at_x[ 1 ] = gamma*x[ 1 ];
    dx = grad_f_obj_at_x;
    cblas_zdscal( n, -1.0, dx.data( ), 1 );
  };

  // Pre-allocations/-calculations.
  vector< complex< double > > grad_f_obj_at_x( n ), dx( n );
  double f_obj0 = f_obj( dx );
  complex< double > one_cplx = { 1.0, 0.0 };

  // Exercise the backtracking line search functionality.
  cout << "Starting point: ( " << x[ 0 ] << ", " << x[ 1 ] << " )" << endl;
  cout << endl;
  cout << "Iteration 0 objective function value: " << f_obj0 << endl;
  for( int i = 0; i < 5; i++ )
  {
    // Assign the gradient and descent direction.
    grad_f_obj( grad_f_obj_at_x, dx );

    /* Backtracking line search followed by taking updating the objective
    function value and taking a step.
    NOTE: dx is overwritten in the backtracking line search function call with
    step_size*dx. */
    double step_size = ase::backtracking_line_search( f_obj, grad_f_obj_at_x, dx, f_obj0 );
    f_obj0 = f_obj( dx );
    cblas_zaxpy( n, &one_cplx, dx.data( ), 1, x.data( ), 1 );

    // Evaluate the new objective function value.
    cout << "Iteration " << i+1 << " objective function value: " << f_obj0 << endl;
  }

  cout << endl;
  cout << "Terminating point: ( " << x[ 0 ] << ", " << x[ 1 ] << " )" << endl;
}
#endif
