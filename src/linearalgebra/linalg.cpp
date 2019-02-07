/*! \file linalg.cpp
    \brief Basic linear algebra routines implemented using Intel MKL functionality.

    Basic linear algebra routines implemented using Intel MKL functionality.
    This functionality includes diagonal matrix product with a general matrix,
    various decompositions, and various methods for computing inverses
    efficiently. Much of the provided functionality is project driven with the
    application being radar signal processing. Thus, where appropriate routines
    exist for handling both real- and complex-valued inputs.
*/

#include <complex>
#include <iostream>
#include <numeric>
#include <vector>
#include "constants/constants.hpp"
#include "mkl.h"

namespace ase
{
namespace linalg
{
  double determinant( const std::vector< double >& X, const double n )
  {
    std::vector< double > X_copy = X;
    std::vector< lapack_int > piv_idx( n );
    LAPACKE_dgetrf( ase::constants::layout, n, n, X_copy.data( ), n, piv_idx.data( ));
    double det = 1.0;
    for( int i = 0; i < n; i++ )
    {
      det *= X_copy[ i+n*i ];

      if(( i+1 ) != piv_idx[ i ]) // Row exchange
        det *= -1.0;

      if( det == 0.0 )
        break;
    }
    return det;
  }

  std::complex< double > determinant( const std::vector< std::complex< double > >& X, const double n )
  {
    std::vector< std::complex< double > > X_copy = X;
    std::vector< lapack_int > piv_idx( n );
    LAPACKE_zgetrf( ase::constants::layout, n, n, X_copy.data( ), n, piv_idx.data( ));
    std::complex< double > det = { 1.0, 0.0 };
    for( int i = 0; i < n; i++ )
    {
      det *= X_copy[ i+n*i ];

      if(( i+1 ) != piv_idx[ i ]) // Row exchange
        det *= -1.0;

      if( det == 0.0 )
        break;
    }
    return det;
  }

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

  void diag_plus_low_rank( const std::vector< double >& x, const std::vector< double >& A,
                           std::vector< double >& B, const bool& transpose_A,
                           const double& alpha, const double& beta )
  {
    int n = x.size( );
    int p = A.size( )/n;

    // Add the low-rank term to B.
    CBLAS_TRANSPOSE trans;
    int lda;
    if( transpose_A )
    {
      trans = CblasTrans;
      lda = p;
    }
    else
    {
      trans = CblasNoTrans;
      lda = n;
    }
    cblas_dsyrk( ase::constants::layout, ase::constants::uplo, trans, n, p,
                 beta, A.data( ), lda, 1.0, B.data( ), n );

    // Add the diagonal components to B.
    for( int i = 0; i < n; i++ )
      B[ i+i*n ] += alpha*x[ i ];
  }

  void diag_plus_low_rank( const std::vector< std::complex< double > >& x,
                           const std::vector< std::complex< double > >& A,
                           std::vector< std::complex< double > >& B,
                           const bool& conj_transpose_A, const double& alpha,
                           const double& beta )
  {
    int n = x.size( );
    int p = A.size( )/n;

    // Add the low-rank term to B.
    CBLAS_TRANSPOSE trans;
    int lda;
    if( conj_transpose_A )
    {
      trans = CblasConjTrans;
      lda = p;
    }
    else
    {
      trans = CblasNoTrans;
      lda = n;
    }
    /* It appears Intel MKL cblas_zherk forces the diagonals to be real. Thus,
    the diagonal update must come after this function call. */
    cblas_zherk( ase::constants::layout, ase::constants::uplo, trans, n, p,
                 beta, A.data( ), lda, 1.0, B.data( ), n );

    // Add the diagonal components to B.
    for( int i = 0; i < n; i++ )
      B[ i+i*n ] += alpha*x[ i ];
  }
} // namespace linalg
} // namespace ase
