/*! \file testComplexRealImaginaryVector.cpp
    \brief Demonstrates utility functions that are useful when working with complex-valued vectors.

    Demonstrates utility functions that are useful when working with
    complex-valued vectors. The specific functions that are
    exercised is/are:
    1. complex_vector()
    2. real_vector()
    3. imag_vector()
*/

#include <iostream>
#include <complex>
#include <vector>
#include "utility/util.hpp"

using namespace std;

#ifndef DOXYGEN_SKIP
int main( int argc, char* argv[ ])
{
  /* Generate separate real and imaginary portions, then construct the
  complex-valued vector from the real and imaginary portions. */
  int n = 5;
  vector< double > x_re( n ), x_im( n );
  cout << "Separate real and imaginary portions:" << endl;
  for( int i = 0; i < n; i++ )
  {
    x_re[ i ] = rand( ) % 20;
    x_im[ i ] = rand( ) % 20;
    cout << "(" << x_re[ i ] << "," << x_im[ i ] << ") ";
  }
  cout << endl;

  // Generate the complex-valued vector.
  vector< complex< double > > x( n );
  ase::util::complex_vector( x_re, x_im, x );
  cout << "\nGenerated complex-valued vector:" << endl;
  for( int i = 0; i < n; i++ )
    cout << x[ i ] << " ";
  cout << endl;

  // Pull out the real portion of x.
  vector< double > x_re2( n );
  ase::util::real_vector( x, x_re2 );
  cout << "\nExtracted real portion of x:" << endl;
  for( int i = 0; i < n; i++ )
    cout << x_re2[ i ] << " ";
  cout << endl;

  // Pull out the imaginary portion of x.
  vector< double > x_im2( n );
  ase::util::imag_vector( x, x_im2 );
  cout << "\nExtracted imaginary portion of x:" << endl;
  for( int i = 0; i < n; i++ )
    cout << x_im2[ i ] << " ";
  cout << endl;
  return 0;
}
#endif
