/*! \file testDiagPlusLowRank.cpp
    \brief Demonstrates the diagonal plus low-rank functionality.

    Demonstrates the diagonal plus low-rank functionality. The specific
    functions that are exercised is/are:
    1. diag_plus_low_rank()
    2. diag_plus_low_rank()
*/

#include <iostream>
#include <complex>
#include <numeric>
#include <vector>
#include "linearalgebra/linalg.hpp"
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
  // Generate a random vector and matrix, and compute the brute force product.
  int n = 2, p = 1;
  vector< double > x( n ), A( n*p ), B_bf( n*n );
  for( int i = 0; i < n; i++ )
  {
    x[ i ] = rand( ) % 20;
    for( int j = 0; j < p; j++ )
      A[ i+n*j ] = rand( ) % 20;
  }

  for( int i = 0; i < n; i++ )
  {
    for( int j = 0; j < i+1; j++ )
    {
      if( i == j )
        B_bf[ i*n+j ] = x[ i ];
      B_bf[ i*n+j ] += A[ j ]*A[ i ];
    }
  }

  // Compute the result using a provided function.
  vector< double > B_fun( n*n );
  ase::linalg::diag_plus_low_rank( x, A, B_fun );

  cout << "Brute force vs. function call" << endl;
  for( int i = 0; i < n*n; i++ )
    cout << B_bf[ i ] << ", " << B_fun[ i ] << endl;
}

void real_valued_example2( )
{
  // Generate a random vector and matrix, and compute the brute force product.
  int n = 2, p = 1;
  vector< double > x( n ), A( n*p ), B_bf( n*n );
  for( int i = 0; i < n; i++ )
  {
    x[ i ] = rand( ) % 20;
    for( int j = 0; j < p; j++ )
      A[ i+n*j ] = rand( ) % 20;
  }

  for( int i = 0; i < n; i++ )
  {
    for( int j = 0; j < i+1; j++ )
    {
      if( i == j )
        B_bf[ i*n+j ] = x[ i ];
      B_bf[ i*n+j ] += A[ j ]*A[ i ];
    }
  }

  // Compute the result using a provided function.
  vector< double > B_fun( n*n );
  ase::linalg::diag_plus_low_rank( x, A, B_fun, true );

  cout << "Brute force vs. function call" << endl;
  for( int i = 0; i < n*n; i++ )
    cout << B_bf[ i ] << ", " << B_fun[ i ] << endl;
}

void complex_valued_example1( )
{
  // Generate a random vector and matrix, and compute the brute force product.
  int n = 2, p = 1;
  vector< complex< double > > x( n ), A( n*p ), B_bf( n*n );
  for( int i = 0; i < n; i++ )
  {
    x[ i ] = {( double )( rand( ) % 20 ), ( double )( rand( ) % 20 )};
    for( int j = 0; j < p; j++ )
      A[ i+n*j ] = {( double )( rand( ) % 20 ), ( double )( rand( ) % 20 )};
  }

  for( int i = 0; i < n; i++ )
  {
    for( int j = 0; j < i+1; j++ )
    {
      if( i == j )
        B_bf[ i*n+j ] = x[ i ];
      B_bf[ i*n+j ] += A[ j ]*conj( A[ i ]);
    }
  }

  // Compute the result using a provided function.
  vector< complex< double > > B_fun( n*n );
  ase::linalg::diag_plus_low_rank( x, A, B_fun );

  cout << "Brute force vs. function call" << endl;
  for( int i = 0; i < n*n; i++ )
    cout << B_bf[ i ] << ", " << B_fun[ i ] << endl;
}

void complex_valued_example2( )
{
  // Generate a random vector and matrix, and compute the brute force product.
  int n = 2, p = 1;
  vector< complex< double > > x( n ), A( n*p ), B_bf( n*n );
  for( int i = 0; i < n; i++ )
  {
    x[ i ] = {( double )( rand( ) % 20 ), ( double )( rand( ) % 20 )};
    for( int j = 0; j < p; j++ )
      A[ i+n*j ] = {( double )( rand( ) % 20 ), ( double )( rand( ) % 20 )};
  }

  for( int i = 0; i < n; i++ )
  {
    for( int j = 0; j < i+1; j++ )
    {
      if( i == j )
        B_bf[ i*n+j ] = x[ i ];
      B_bf[ i*n+j ] += A[ j ]*conj( A[ i ]);
    }
  }

  // Compute the result using a provided function.
  vector< complex< double > > B_fun( n*n );
  ase::linalg::diag_plus_low_rank( x, A, B_fun, true );

  cout << "Brute force vs. function call" << endl;
  for( int i = 0; i < n*n; i++ )
    cout << B_bf[ i ] << ", " << B_fun[ i ] << endl;
}
#endif
