/*! \file testDeterminant.cpp
    \brief Demonstrates determinant functionality.

    Demonstrates determinant functionality. The specific functions that are
    exercised is/are:
    1. determinant()
    2. determinant()
    3. determinant_diag_plus_low_rank()
    4. determinant_diag_plus_low_rank()
*/

#include <iostream>
#include <complex>
#include <numeric>
#include <vector>
#include "linearalgebra/linalg.hpp"
#include "mkl.h"

using namespace std;

#ifndef DOXYGEN_SKIP
void real_valued_example( );
void complex_valued_example( );

int main( int argc, char* argv[ ])
{
  // Real-valued examples.
  cout << "Example Result #1 Using Real-Valued Arguments" << endl;
  cout << "-----------------------------------------------------------" << endl;
  cout << endl;
  srand( 0 );
  real_valued_example( );
  cout << endl;

  // Complex-valued examples.
  cout << "Example Result #1 Using Complex-Valued Arguments" << endl;
  cout << "-----------------------------------------------------------" << endl;
  cout << endl;
  srand( 0 );
  complex_valued_example( );
  cout << endl;

  return 0;
}

void real_valued_example( )
{
  int n = 2;
  vector< double > X( n*n );
  for( int i = 0; i < n*n; i++ )
    X[ i ] = rand( ) % 10;

  // Brute force determinant.
  double det_bf = X[ 0 ]*X[ 3 ]-X[ 1 ]*X[ 2 ];
  double det = ase::linalg::determinant( X, n );
  cout << "Brute force determinant: " << det_bf << endl;
  cout << "Determinant function: " << det << endl;
}

void complex_valued_example( )
{
  int n = 2;
  vector< complex< double > > X( n*n );
  for( int i = 0; i < n*n; i++ )
    X[ i+i*n ] = {( double )( rand( ) % 10 ), ( double )( rand( ) % 10 )};

  // Brute force determinant.
  complex< double > det_bf = X[ 0 ]*X[ 3 ]-X[ 1 ]*X[ 2 ];
  complex< double > det = ase::linalg::determinant( X, n );
  cout << "Brute force determinant: " << det_bf << endl;
  cout << "Determinant function: " << det << endl;
}
#endif
