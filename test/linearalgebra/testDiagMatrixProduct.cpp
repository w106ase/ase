/*! \file testDiagMatrixProduct.cpp
    \brief Demonstrates the diagonal matrix product with a general matrix functionality.

    Demonstrates the diagonal matrix product with a general matrix
    functionality. The specific functions that are exercised is/are:
    1. diag_matrix_product()
    2. diag_matrix_product()
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
  int n = 5, p = 2;
  vector< double > x( n ), A( n*p ), B_bf( n*p );
  for( int i = 0; i < n; i++ )
  {
    x[ i ] = rand( ) % 20;
    cout << x[ i ] << ", ";
    for( int j = 0; j < p; j++ )
    {
      A[ i+n*j ] = rand( ) % 20;
      B_bf[ i+n*j ] = x[ i ]*A[ i+n*j ];
      cout << A[ i+n*j ] << " ";
    }
    cout << endl;
  }
  cout << endl;

  // Compute the result using a provided function.
  vector< double > B_fun( n*p );
  ase::linalg::diag_matrix_product( x, A, B_fun );

  cout << "Brute force vs. function call" << endl;
  for( int i = 0; i < n*p; i++ )
    cout << B_bf[ i ] << ", " << B_fun[ i ] << endl;
}

void real_valued_example2( )
{
  // Generate a random vector and matrix, and compute the brute force product.
  int n = 5, p = 2;
  vector< double > x( n ), A( n*p ), B_bf( n*p );
  for( int i = 0; i < p; i++ )
  {
    x[ i ] = rand( ) % 23;
    cout << x[ i ] << ", ";
    for( int j = 0; j < n; j++ )
    {
      A[ n*i+j ] = rand( ) % 23;
      B_bf[ n*i+j ] = x[ i ]*A[ n*i+j ];
      cout << A[ n*i+j ] << " ";
    }
    cout << endl;
  }
  cout << endl;

  // Compute the result using a provided function.
  vector< double > B_fun( n*p );
  ase::linalg::diag_matrix_product( x, A, B_fun, 1.0, false );

  cout << "Brute force vs. function call" << endl;
  for( int i = 0; i < n*p; i++ )
    cout << B_bf[ i ] << ", " << B_fun[ i ] << endl;
}

void complex_valued_example1( )
{
  // Generate a random vector and matrix, and compute the brute force product.
  int n = 5, p = 2;
  double x_re, x_im;
  vector< complex< double > > x( n ), A( n*p ), B_bf( n*p );
  for( int i = 0; i < n; i++ )
  {
    x_re = rand( ) % 20;
    x_im = rand( ) % 20;
    x[ i ] = { x_re, x_im };
    cout << x[ i ] << ", ";
    for( int j = 0; j < p; j++ )
    {
      x_re = rand( ) % 20;
      x_im = rand( ) % 20;
      A[ i+n*j ] = { x_re, x_im };
      B_bf[ i+n*j ] = x[ i ]*A[ i+n*j ];
      cout << A[ i+n*j ] << " ";
    }
    cout << endl;
  }
  cout << endl;

  // Compute the result using a provided function.
  vector< complex< double > > B_fun( n*p );
  ase::linalg::diag_matrix_product( x, A, B_fun );

  cout << "Brute force vs. function call" << endl;
  for( int i = 0; i < n*p; i++ )
    cout << B_bf[ i ] << ", " << B_fun[ i ] << endl;
}

void complex_valued_example2( )
{
  // Generate a random vector and matrix, and compute the brute force product.
  int n = 5, p = 2;
  double x_re, x_im;
  vector< complex< double > > x( n ), A( n*p ), B_bf( n*p );
  for( int i = 0; i < p; i++ )
  {
    x_re = rand( ) % 20;
    x_im = rand( ) % 20;
    x[ i ] = { x_re, x_im };
    cout << x[ i ] << ", ";
    for( int j = 0; j < n; j++ )
    {
      x_re = rand( ) % 20;
      x_im = rand( ) % 20;
      A[ n*i+j ] = { x_re, x_im };
      B_bf[ n*i+j ] = x[ i ]*A[ n*i+j ];
      cout << A[ n*i+j ] << " ";
    }
    cout << endl;
  }
  cout << endl;

  // Compute the result using a provided function.
  vector< complex< double > > B_fun( n*p );
  ase::linalg::diag_matrix_product( x, A, B_fun, 1.0, false );

  cout << "Brute force vs. function call" << endl;
  for( int i = 0; i < n*p; i++ )
    cout << B_bf[ i ] << ", " << B_fun[ i ] << endl;
}
#endif
