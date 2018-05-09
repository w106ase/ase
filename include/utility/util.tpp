/*! \file util.tpp
    \brief Utility routines used for data wrangling, testing, and verification.

    Utility routines used for data wrangling, testing, and verification. For
    example, this header contains functions for computing the real/imaginary
    part of a complex-valued vector, as well as various routines for using
    MATLAB engine within C++ for testing, verifying, and visualizing results.
*/

#include <algorithm>
#include <complex>
#include <functional>
#include <vector>

namespace ase
{
namespace util
{
  template< class T >
  void complex_vector( const std::vector< T >& x_re, const std::vector< T >& x_im,
                       std::vector< std::complex< T > >& x )
  {
    transform( x_re.begin( ), x_re.end( ), x_im.begin( ), x.begin( ),
               [ ]( T x_re, T x_im ) -> std::complex< T > { return ( std::complex< T > ){ x_re, x_im }; });
  }

  template< class T >
  void imag_vector( const std::vector< std::complex< T > >& x, std::vector< T >& x_im )
  {
    transform( x.begin( ), x.end( ), x_im.begin( ),
               [ ]( std::complex< T > x ) -> T { return imag( x ); });
  }

  template< class T >
  void real_vector( const std::vector< std::complex< T > >& x, std::vector< T >& x_re )
  {
    transform( x.begin( ), x.end( ), x_re.begin( ),
               [ ]( std::complex< T > x ) -> T { return real( x ); });
  }
} // namespace util
} // namespace ase
