/*! \file testGeneralBarrierMethod.cpp
    \brief Demonstrates the general barrier method.
*/

#include <iostream>
#include <complex>
#include <vector>
#include "convexoptimization/cvxopt.hpp"
#include "mkl.h"

using namespace std;

#ifndef DOXYGEN_SKIP
void real_valued_example( );

int main( int argc, char* argv[ ])
{
  // Real-valued example.
  cout << "Example Result Using Real-Valued Arguments" << endl;
  cout << "-----------------------------------------------------------" << endl;
  cout << endl;
  srand( 0 );
  real_valued_example( );
  cout << endl;

  return 0;
}

void real_valued_example( )
{
  /* Specify dimension of the problem, generate data, and a strictly feasible
  initial point. */
  int n = 5;
  vector< double > c( n ), x( n, 1.0 );
  cout << "c = ";
  for( int i = 0; i < n; i++ )
  {
      c[ i ] = rand( ) % 13;
      cout << c[ i ] << " ";
  }
  cout << endl;
  cout << "x = ";
  for( int i = 0; i < n; i++ )
      cout << x[ i ] << " ";
  cout << endl;

  cout << "Initial objective function value: " << cblas_ddot( n, x.data( ), 1, c.data( ), 1 ) << endl;
  cout << endl;


  // Define a lambda for the centering function.
  auto f_center = [ &c, &n ]( vector< double >& x, const double& barrier_parameter ) -> bool
  {
    // Define a lambda for the barrier objective function.
    auto f_barrier_obj = [ &c, &barrier_parameter, &n ]( const vector< double >& x ) -> double
    {
      // Compute the objective function.
      return barrier_parameter*cblas_ddot( n, x.data( ), 1, c.data( ), 1 )+ase::cvx::log_barrier( x, true );
    };

    // Define a lambda for the gradient of the barrier objective function.
    auto grad_f_barrier_obj = [ &c, &barrier_parameter, &n ]( const vector< double >& x,
                                                              vector< double >& grad_f_barrier_obj_at_x ) -> void
    {
      for( int i = 0; i < n; i++ )
        grad_f_barrier_obj_at_x[ i ] = barrier_parameter*c[ i ]-1.0/x[ i ];
    };

    // Define a lambda for the barrier descent direction.
    auto barrier_desc_dir = [ &n ]( const vector< double >& x,
                                    const vector< double >& grad_f_barrier_obj_at_x,
                                    vector< double >& dx ) -> void
    {
      dx = grad_f_barrier_obj_at_x;
      cblas_dscal( n, -1.0, dx.data( ), 1 );
    };

    // Use general descent method to solve the unconstrained barrier problem.
    return ase::cvx::general_descent_method_with_btls( f_barrier_obj, grad_f_barrier_obj,
                                                       barrier_desc_dir, x, 100, pow( 10, -6.0 ),
                                                       0.45, 0.8, 500 );
  };

  // Exercise the general barrier method function.
  ase::cvx::general_barrier_method( f_center, x );

  cout << "x = ";
  for( int i = 0; i < n; i++ )
      cout << x[ i ] << " ";
  cout << endl;
  cout << "Ending objective function value: " << cblas_ddot( n, x.data( ), 1, c.data( ), 1 ) << endl;
  cout << endl;
}
#endif
