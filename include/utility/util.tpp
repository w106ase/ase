/*! \file cvxopt.cpp
    \brief Convex optimization functionality implemented using Intel MKL
    routines.

    Convex optimization functionality using Intel MKL functionality. This
    functionality includes line search routines, descent methods, and various
    generic and specialized solvers for convex optimization problems. Much of
    the provided functionality is project driven with the application being
    radar signal processing. Thus, where appropriate routines exist for handling
    both real- and complex-valued inputs.
*/

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
