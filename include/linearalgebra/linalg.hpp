/*! \file linalg.hpp
    \brief Basic linear algebra routines implemented using Intel MKL functionality.

    Basic linear algebra routines implemented using Intel MKL functionality.
    This functionality includes diagonal matrix product with a general matrix,
    various decompositions, and various methods for computing inverses
    efficiently. Much of the provided functionality is project driven with the
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
  /*! \brief Compute the determinant of a square, real-valued matrix.

      \param X square, real-valued matrix to compute determinant for.
      \param n number of rows and columns in <tt>X</tt>.

      \return determinant of <tt>X</tt>.
  */
  double determinant( const std::vector< double >& X, const double n );

  /*! \brief Compute the determinant of a square, Hermitian matrix.

      \param X square, Hermitian matrix to compute determinant for.
      \param n number of rows and columns in <tt>X</tt>.

      \return determinant of <tt>X</tt>.
  */
  std::complex< double > determinant( const std::vector< std::complex< double > >& X, const double n );

  /*! \brief Compute the determinant of a sum of a matrix plus a diagonal plus low-rank term (i.e., \f$B+\alpha \mathrm{Diag}(x)+\beta A A^{\mathrm{T}}\f$ or \f$B+\alpha \mathrm{Diag}(x)+\beta A^{\mathrm{T}} A\f$).

      \param x diagonal elements forming the diagonal matrix.
      \param A low-rank term.
      \param transpose_A flag which is true when computing the low-rank term as \f$ A^{\mathrm{T}} A\f$.
      \param alpha scaling for the diagonal term.
      \param beta scaling for the low-rank term.

      \return determinant of \f$\alpha \mathrm{Diag}(x)+\beta A
      A^{\mathrm{T}}\f$ or \f$\alpha \mathrm{Diag}(x)+\beta A^{\mathrm{T}}
      A\f$).
  */
  double determinant_diag_plus_low_rank( const std::vector< double >& x,
                                         const std::vector< double >& A,
                                         const bool& conj_transpose_A = false,
                                         const double& alpha = 1.0, const double& beta = 1.0 );

  /*! \brief Real-valued diagonal matrix product with an arbitrary real-valued matrix (i.e., \f$ \alpha \mathrm{Diag}(x) A + B\f$ or \f$ \alpha A \mathrm{Diag}(x) + B\f$).

      \param x diagonal elements forming the diagonal matrix.
      \param A matrix to be multiplied with a diagonal matrix.
      \param B matrix to store the matrix product.
      \param alpha scalar out front of the matrix product.
      \param multiply_diag_on_lhs flag that is true when computing \f$ \alpha \mathrm{Diag}(x) A \f$ (when false it computes \f$ \alpha A \mathrm{Diag}(x) + B\f$)
  */
  void diag_matrix_product( const std::vector< double >& x,
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
  void diag_matrix_product( const std::vector< std::complex< double > >& x,
                            const std::vector< std::complex< double > >& A,
                            std::vector< std::complex< double > >& B,
                            const std::complex< double >& alpha = 1.0,
                            const bool& multiply_diag_on_lhs = true );

  /*! \brief Compute the sum of a matrix plus a diagonal plus low-rank term (i.e., \f$B+\alpha \mathrm{Diag}(x)+\beta A A^{\mathrm{T}}\f$ or \f$B+\alpha \mathrm{Diag}(x)+\beta A^{\mathrm{T}} A\f$).

      \param x diagonal elements forming the diagonal matrix.
      \param A low-rank term.
      \param B matrix to be added to the diagonal plus low-rank result.
      \param transpose_A flag which is true when computing the low-rank term as \f$ A^{\mathrm{T}} A\f$.
      \param alpha scaling for the diagonal term.
      \param beta scaling for the low-rank term.

      \note Because <tt>B</tt> is added to the diagonal plus low-rank result,
      the user should ensure <tt>B</tt> contains all zeros when looking for
      assignment (i.e., to compute i.e., \f$B = \alpha \mathrm{Diag}(x)+\beta A
      A^{\mathrm{T}}\f$ or \f$B = \alpha \mathrm{Diag}(x)+\beta A^{\mathrm{T}}
      A\f$).
  */
  void diag_plus_low_rank( const std::vector< double >& x, const std::vector< double >& A,
                           std::vector< double >& B, const bool& conj_transpose_A = false,
                           const double& alpha = 1.0, const double& beta = 1.0 );

  /*! \brief Compute the sum of a matrix plus a diagonal plus low-rank term (i.e., \f$B+\alpha \mathrm{Diag}(x)+\beta A A^{\mathrm{H}}\f$ or \f$B+\alpha \mathrm{Diag}(x)+\beta A^{\mathrm{H}} A\f$).

      \param x diagonal elements forming the diagonal matrix.
      \param A low-rank term.
      \param B matrix to be added to the diagonal plus low-rank result.
      \param conj_transpose_A flag which is true when computing the low-rank term as \f$ A^{\mathrm{H}} A\f$.
      \param alpha scaling for the diagonal term.
      \param beta scaling for the low-rank term.

      \note Because <tt>B</tt> is added to the diagonal plus low-rank result,
      the user should ensure <tt>B</tt> contains all zeros when looking for
      assignment (i.e., to compute i.e., \f$B = \alpha \mathrm{Diag}(x)+\beta A
      A^{\mathrm{H}}\f$ or \f$B = \alpha \mathrm{Diag}(x)+\beta A^{\mathrm{H}}
      A\f$).
  */
  void diag_plus_low_rank( const std::vector< std::complex< double > >& x,
                           const std::vector< std::complex< double > >& A,
                           std::vector< std::complex< double > >& B,
                           const bool& conj_transpose_A = false,
                           const double& alpha = 1.0, const double& beta = 1.0 );
} // namespace linalg
} // namespace ase
#endif
