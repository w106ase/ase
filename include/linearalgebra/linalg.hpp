/*! \file linalg.hpp
    \brief Basic linear algebra routines implemented using Intel MKL functionality.

    Basic linear algebra routines implemented using Intel MKL functionality.
    This functionality includes diagonal matrix product with a general matrix,
    various decompositions, and various methods for computing inverses
    efficiently.  Much of the provided functionality is project driven with the
    application being radar signal processing. Thus, where appropriate routines
    exist for handling both real- and complex-valued inputs.
*/

#ifndef LINALG_H
#define LINALG_H
#include <complex>
#include <vector>

namespace ase
{
namespace linalg
{
  /*! \brief Real-valued diagonal matrix product with an arbitrary real-valued matrix (i.e., \f$ \alpha \mathrm{Diag}(x) A + B\f$ or \f$ \alpha A \mathrm{Diag}(x) + B\f$).

      \param x diagonal elements forming the diagonal matrix.
      \param A matrix to be multiplied with a diagonal matrix.
      \param B matrix to store the matrix product.
      \param alpha scalar out front of the matrix product.
      \param multiply_diag_on_lhs flag that is true when computing \f$ \alpha \mathrm{Diag}(x) A \f$ (when false it computes \f$ \alpha A \mathrm{Diag}(x) + B\f$)
  */
  void diag_matrix_product( std::vector< double >& x,
                            const std::vector< double >& A,
                            std::vector< double >& B,
                            const double& alpha = 1.0,
                            const bool& multiply_diag_on_lhs = true );

  /*! \brief Complex-valued diagonal matrix product with an arbitrary complex-valued matrix (i.e., \f$ \alpha \mathrm{Diag}(x) A + B\f$ or \f$ \alpha A \mathrm{Diag}(x) + B\f$).

      \param x diagonal elements forming the diagonal matrix.
      \param A matrix to be multiplied with a diagonal matrix.
      \param B matrix to store the matrix product.
      \param alpha scalar out front of the matrix product.
      \param multiply_diag_on_lhs flag that is true when computing \f$ \alpha \mathrm{Diag}(x) A \f$ (when false it computes \f$ \alpha A \mathrm{Diag}(x) + B\f$)
  */
  void diag_matrix_product( std::vector< std::complex< double > >& x,
                            const std::vector< std::complex< double > >& A,
                            std::vector< std::complex< double > >& B,
                            const std::complex< double >& alpha = 1.0,
                            const bool& multiply_diag_on_lhs = true );
} // namespace linalg
} // namespace ase
#endif
