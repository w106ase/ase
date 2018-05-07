/*! \file util.hpp
    \brief Utility routines used for data wrangling, testing, and verification.

    Utility routines used for data wrangling, testing, and verification. For
    example, this header contains functions for computing the real/imaginary
    part of a complex-valued vector, as well as various routines for using
    MATLAB engine within C++ for testing, verifying, and visualizing results.
*/

#ifndef UTIL_H
#define UTIL_H
#include <complex>
#include <vector>

namespace ase
{
namespace util
{
  /*! \brief Generates a complex-valued vector given vectors containing the real and imaginary parts.

      \param x_re vector containing the real values for <tt>x</tt>.
      \param x_im vector containing the imaginary values for <tt>x</tt>.
      \param x vector to hold the complex-values produced from <tt>x_re</tt> and <tt>x_im</tt>.
  */
  template< class T >
  void complex_vector( const std::vector< T >& x_re, const std::vector< T >& x_im,
                       std::vector< std::complex< T > >& x );

  /*! \brief Generates a real-valued vector containing the imaginary portion of a given complex-valued vector.

      \param x complex-valued vector whose real portion is to be returned.
      \param x_im vector to hold the imaginary portion of <tt>x</tt>.
  */
  template< class T >
  void imag_vector( const std::vector< std::complex< T > >& x, std::vector< T >& x_im );

  /*! \brief Generates a real-valued vector containing the real portion of a given complex-valued vector.

      \param x complex-valued vector whose real portion is to be returned.
      \param x_re vector to hold the real portion of <tt>x</tt>.
  */
  template< class T >
  void real_vector( const std::vector< std::complex< T > >& x, std::vector< T >& x_re );
} // namespace util
} // namespace ase
#include "utility/util.tpp"
#endif
