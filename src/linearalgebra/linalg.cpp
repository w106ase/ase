/*! \file linalg.cpp
    \brief Basic linear algebra routines implemented using Intel MKL functionality.

    Basic linear algebra routines implemented using Intel MKL functionality.
    This functionality includes diagonal matrix product with a general matrix,
    various decompositions, and various methods for computing inverses
    efficiently.  Much of the provided functionality is project driven with the
    application being radar signal processing. Thus, where appropriate routines
    exist for handling both real- and complex-valued inputs.
*/

#include <complex>
#include <iostream>
#include <numeric>
#include <vector>
#include "mkl.h"

namespace ase
{
namespace linalg
{
  void diag_matrix_product( const std::vector< double >& x,
                            const std::vector< double >& A,
                            std::vector< double >& B,
                            const double& alpha,
                            const bool& multiply_diag_on_lhs )
  {
    int n_rows = x.size( );
    int n_cols = A.size( )/n_rows;

    if( multiply_diag_on_lhs )
    {
      for( int i = 0; i < n_rows; i++ )
        cblas_daxpy( n_cols, x[ i ], &A[ i ], n_rows, &B[ i ], n_rows );
    }
    else
    {
      for( int i = 0; i < n_cols; i++ )
        cblas_daxpy( n_rows, x[ i ], &A[ i*n_rows ], 1, &B[ i*n_rows ], 1 );
    }
  }

  void diag_matrix_product( const std::vector< std::complex< double > >& x,
                            const std::vector< std::complex< double > >& A,
                            std::vector< std::complex< double > >& B,
                            const std::complex< double >& alpha,
                            const bool& multiply_diag_on_lhs )
  {
    int n_rows = x.size( );
    int n_cols = A.size( )/n_rows;

    if( multiply_diag_on_lhs )
    {
      for( int i = 0; i < n_rows; i++ )
        cblas_zaxpy( n_cols, &x[ i ], &A[ i ], n_rows, &B[ i ], n_rows );
    }
    else
    {
      for( int i = 0; i < n_cols; i++ )
        cblas_zaxpy( n_rows, &x[ i ], &A[ i*n_rows ], 1, &B[ i*n_rows ], 1 );
    }
  }
} // namespace linalg
} // namespace ase
