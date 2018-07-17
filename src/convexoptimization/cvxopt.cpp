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

#include <algorithm>
#include <complex>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "constants/constants.hpp"
#include "utility/util.hpp"
#include "linearalgebra/linalg.hpp"
#include "convexoptimization/cvxopt.hpp"
#include "mkl.h"

namespace ase
{
namespace cvx
{
  double backtracking_line_search( const std::function< double ( const std::vector< double >& x ) >& f_obj,
                                   std::vector< double >& x, const std::vector< double >& grad_f_obj_at_x, std::vector< double >& dx,
                                   const double& alpha, const double& beta, const int& max_iter )
  {
    // Pre-allocations/-calculations.
    std::vector< double > x0 = x; // Retain copy of the starting point
    double step_size = 1.0, f_obj0 = f_obj( x );
    int iter = 0, n = dx.size( );
    double grad_f_obj_dx = cblas_ddot( n, grad_f_obj_at_x.data( ), 1, dx.data( ), 1 );

    // Update x using a unit step-size and evaluate the objective function.
    cblas_daxpy( n, step_size, dx.data( ), 1, x.data( ), 1 );
    double f_obj_new = f_obj( x );

    while( f_obj_new > ( f_obj0+step_size*alpha*grad_f_obj_dx ))
    {
      // Break if maximum number of iterations is reached.
      if( iter == max_iter )
      {
        std::cout << "Warning: Backtracking line search reached maximum number "
                  << "of iterations -- step-size is being set to zero." << std::endl;
        std::cout << std::endl;
        x = x0;
        step_size = 0.0;
        break;
      }

      // Evaluate the objective function using the current step-size.
      step_size *= beta;
      x = x0;
      cblas_daxpy( n, step_size, dx.data( ), 1, x.data( ), 1 );
      f_obj_new = f_obj( x );

      // Increment the counter.
      iter += 1;
    }

    return step_size;
  }

  double backtracking_line_search( const std::function< double ( const std::vector< std::complex< double > >& x ) >& f_obj,
                                   std::vector< std::complex< double > >& x, const std::vector< std::complex< double > >& grad_f_obj_at_x,
                                   std::vector< std::complex< double > >& dx, const double& alpha, const double& beta, const int& max_iter )
  {
    // Pre-allocations/-calculations.
    std::vector< std::complex< double > > x0 = x; // Retain copy of the starting point
    double step_size = 1.0, f_obj0 = f_obj( x );
    int iter = 0, n = dx.size( );

    /* NOTE: this quantity should always be real-valued. To enforce this only
    the real portion is retained. */
    std::complex< double > grad_f_obj_dx_cplx;
    cblas_zdotc_sub( n, grad_f_obj_at_x.data( ), 1, dx.data( ), 1, &grad_f_obj_dx_cplx );
    double grad_f_obj_dx = real( grad_f_obj_dx_cplx );

    // Update x using a unit step-size and evaluate the objective function.
    cblas_zaxpy( n, &step_size, dx.data( ), 1, x.data( ), 1 );
    double f_obj_new = f_obj( x );

    while( f_obj_new > ( f_obj0+step_size*alpha*grad_f_obj_dx ))
    {
      // Break if maximum number of iterations is reached.
      if( iter == max_iter )
      {
        std::cout << "Warning: Backtracking line search reached maximum number "
                  << "of iterations -- step-size is being set to zero." << std::endl;
        std::cout << std::endl;
        x = x0;
        step_size = 0.0;
        break;
      }

      // Evaluate the objective function using the current step-size.
      step_size *= beta;
      x = x0;
      cblas_zaxpy( n, &step_size, dx.data( ), 1, x.data( ), 1 );
      f_obj_new = f_obj( x );

      // Increment the counter.
      iter += 1;
    }
    return step_size;

  }

  bool general_descent_method_with_btls( const std::function< double ( const std::vector< double >& x ) >& f_obj,
                                         const std::function< void ( const std::vector< double >& x, std::vector< double >& grad_f_obj_at_x ) >& grad_f_obj,
                                         const std::function< void ( const std::vector< double >& x, const std::vector< double >& grad_f_obj_at_x, std::vector< double >& dx ) >& desc_dir,
                                         std::vector< double >& x, const int& max_iter, const double& norm2_grad_thresh,
                                         const double& rel_dx_norm2_thresh, const double& alpha, const double& beta, const int& max_btls_iter )
  {
    // Pre-allocations/-calculations.
    int iter = 0, n = x.size( );
    double step_size, norm2_x = cblas_dnrm2( n, x.data( ), 1 );
    bool process_halted = false;
    std::vector< double > grad_f_obj_at_x( n ), dx( n );
    double rel_dx = std::numeric_limits< double >::infinity( );

    while( iter < max_iter )
    {
      // Compute the gradient at the current point and a descent direction.
      grad_f_obj( x, grad_f_obj_at_x );
      desc_dir( x, grad_f_obj_at_x, dx );

      // Check the stopping criterion.
      if( cblas_dnrm2( n, grad_f_obj_at_x.data( ), 1 ) < norm2_grad_thresh ||
          rel_dx < rel_dx_norm2_thresh )
        break;

      /* Backtracking line search.
      NOTE: x is overwritten in the function call with x+step_size*dx. */
      step_size = backtracking_line_search( f_obj, x, grad_f_obj_at_x, dx, alpha, beta, max_btls_iter );

      /* If a step-size of zero is encountered, the backtracking line search is
      hitting its maximum number of iterations and no more progress will be made. */
      if( step_size == 0.0 )
      {
        process_halted = true;
        rel_dx = 0.0;
        break;
      }
      else
      {
        if( norm2_x == 0.0 )
          norm2_x = cblas_dnrm2( n, x.data( ), 1 );

        rel_dx = step_size*cblas_dnrm2( n, dx.data( ), 1 )/norm2_x;
      }

      // Increment the counter.
      iter += 1;
    }

    return process_halted;
  }

  bool general_descent_method_with_btls( const std::function< double ( const std::vector< std::complex< double > >& x ) >& f_obj,
                                         const std::function< void ( const std::vector< std::complex< double > >& x, std::vector< std::complex< double > >& grad_f_obj_at_x ) >& grad_f_obj,
                                         const std::function< void ( const std::vector< std::complex< double > >& x, const std::vector< std::complex< double > >& grad_f_obj_at_x, std::vector< std::complex< double > >& dx ) >& desc_dir,
                                         std::vector< std::complex< double > >& x, const int& max_iter, const double& norm2_grad_thresh,
                                         const double& rel_dx_norm2_thresh, const double& alpha, const double& beta, const int& max_btls_iter )
  {
    // Pre-allocations/-calculations.
    int iter = 0, n = x.size( );
    double step_size, norm2_x;
    bool process_halted = false;
    std::vector< std::complex< double > > grad_f_obj_at_x( n ), dx( n );
    double rel_dx = std::numeric_limits< double >::infinity( );

    while( iter < max_iter )
    {
      // Compute the gradient at the current point and a descent direction.
      grad_f_obj( x, grad_f_obj_at_x );
      desc_dir( x, grad_f_obj_at_x, dx );

      // Check the stopping criterion.
      if( cblas_dznrm2( n, grad_f_obj_at_x.data( ), 1 ) < norm2_grad_thresh )
        break;

      /* Backtracking line search.
      NOTE: x is overwritten in the function call with x+step_size*dx. */
      step_size = backtracking_line_search( f_obj, x, grad_f_obj_at_x, dx, alpha, beta, max_btls_iter );

      /* If a step-size of zero is encountered, the backtracking line search is
      hitting its maximum number of iterations and no more progress will be made. */
      if( step_size == 0.0 )
      {
        process_halted = true;
        rel_dx = 0.0;
        break;
      }
      else
      {
        if( norm2_x == 0.0 )
          norm2_x = cblas_dznrm2( n, x.data( ), 1 );

        rel_dx = step_size*cblas_dznrm2( n, dx.data( ), 1 )/norm2_x;
      }

      // Increment the counter.
      iter += 1;
    }

    return process_halted;
  }

  void general_barrier_method( const std::function< bool ( std::vector< double >& x, const double& barrier_parameter ) >& f_center,
                               std::vector< double >& x, const double& barrier_parameter0, const double& barrier_parameter_update,
                               const int& max_iter, const double& sub_optimality_thresh )
  {
    // Pre-allocations/-calculations.
    int iter = 0, n = x.size( );
    bool process_halted = false;
    double barrier_parameter = barrier_parameter0;

    while( iter < max_iter )
    {
        // Centering step.
        process_halted = f_center( x, barrier_parameter );

        // Evaluate the sub-optimality.
        if( n/barrier_parameter < sub_optimality_thresh || process_halted )
          break;

        // Increase the barrier parameter and increment the counter.
        barrier_parameter *= barrier_parameter_update;
        iter += 1;
    }
  }

  double log_barrier( const std::vector< double >& x, const bool& negate_x )
  {
    if( std::all_of( x.begin( ), x.end( ),
        [ &negate_x ]( const double& x ){ return ( negate_x ) ? ( x > 0.0 ) : ( x < 0.0 ); }))
    {
      // Pre-allocations/-calculations.
      int n = x.size( );
      std::vector< double > ln_x( n );

      // Negate x if needed and compute the natural logarithm.
      if( negate_x ) // Compute ln( -[-x])
        vdLn( n, x.data( ), ln_x.data( ));
      else // Compute ln( -x )
      {
        std::vector< double > x_copy = x;
        cblas_dscal( n, -1.0, x_copy.data( ), 1 );
        vdLn( n, x_copy.data( ), ln_x.data( ));
      }

      // Compute the negative of the sum of the natural logarithm.
      std::vector< double > ones( n, 1.0 );
      return -cblas_ddot( n, ln_x.data( ), 1, ones.data( ), 1 );
    }
    else
      return std::numeric_limits< double >::infinity( );
  }

  void sdp_inequality_form_with_diag_plus_low_rank_lmi( std::vector< double >&x,
                                                        const std::vector< double >& c,
                                                        std::vector< double >& Z,
                                                        std::vector< double >& lmi_inv,
                                                        const double& precision )
  {
    if( precision >= 1.0 || precision <= 0.0 )
      throw std::invalid_argument( "Precision value must be in the interval (0, 1)." );

    // Input dimension.
    int n = x.size( ), p = Z.size( )/n, n2 = n*n, p2 = p*p, np = n*p, np2 = n*p2, p4 = p2*p2;
    double neg_log_det_C;
    bool use_alt_newton_step = p2 < 0.2*n;
    std::vector< double > V( p2 ), hess( n2 ), S( np );

    // Initialize a strictly feasible point x.
    double Z_fro_norm = LAPACKE_dlange( ase::constants::layout, 'F', n, p, Z.data( ), n );
    cblas_dscal( np, 1.0/Z_fro_norm, Z.data( ), 1 );
    x.assign( n, -( 1.0+1.0e-6 ));

    /* Define a lambda for computing the eigendecomposition of
    C = eye( p )+Z^{T} Diag^{-1}(x) Z, so that V := V Diag^{-1/2}(w) where w
    contains the eigenvalues of C and V are the eigenvectors of C. */
    auto f_C = [ &n, &p, &p2 ]( const std::vector< double >& x,
                                const std::vector< double >& Z,
                                std::vector< double >& V,
                                double& neg_log_det_C ) -> double
    {
      // Initialize C with the identity matrix.
      std::vector< double > C( p2 );
      for( int i = 0; i < p; i++ )
        C[ i+i*p ] = 1.0;

      // Add the term Z^{T} Diag^{-1}(x) Z.
      for( int i = 0; i < n; i++ )
        cblas_dsyr( ase::constants::layout, ase::constants::uplo,
                    p, 1.0/x[ i ], &Z[ i ], n, C.data( ), p );

      // Compute eigendecomposition of C = V W V^{T}.
      std::vector< double > w( p );
      MKL_INT m;
      std::vector< MKL_INT > isuppz( 2*p );
      LAPACKE_dsyevr( ase::constants::layout, 'V', 'A', ase::constants::uplo_char,
                      p, C.data( ), p, 0.0, 0.0, 0.0, 0.0, LAPACKE_dlamch( 'S' ),
                      &m, w.data( ), V.data( ), p, isuppz.data( ));

      // Compute V := V Diag^{-1/2}(w)
      std::vector< double > w_inv_sqrt( p );
      vdInvSqrt( p, w.data( ), w_inv_sqrt.data( ));
      C.assign( p2, 0.0 );
      ase::linalg::diag_matrix_product( w_inv_sqrt, V, C, 1.0, false );
      V = C;

      // Compute the negative logarithm of the determinant of C.
      neg_log_det_C = 0.0;
      for( int i = 0; i < p; i++ )
      {
        if( w[ i ] > 0.0 )
          neg_log_det_C -= log( w[ i ]);
        else
        {
          neg_log_det_C = std::numeric_limits< double >::infinity( );
          break;
        }
      }
      return neg_log_det_C;
    };

    // Define a lambda for the f_center function.
    auto f_center = [ &c, &Z, &f_C, &V, &neg_log_det_C, &S, &use_alt_newton_step,
                      &lmi_inv, &hess, &precision, &n, &p, &np, &n2, &p2, &np2, &p4 ]( std::vector< double >& x,
                                                                           const double& barrier_parameter ) -> bool
    {
      // Compute the eigendecomposition of C = eye( p )+Z^{T} Diag^{-1}(x) Z.
      f_C( x, Z, V, neg_log_det_C );
      bool call_f_C;

      // Define a lambda for the gradient of the barrier objective function.
      auto grad_f_barrier_obj = [ &c, &Z, &barrier_parameter, &V, &call_f_C, &S,
                                  &lmi_inv, &n, &p, &np ]( const std::vector< double >& x,
                                                           std::vector< double >& grad_f_barrier_obj_at_x ) -> void
      {
        // Compute S = Diag^{-1}(x) Z V, where V := V Diag^{-1/2}(w).
        std::vector< double > x_inv( n ), diag_x_inv_Z( np );
        vdInv( n, x.data( ), x_inv.data( ));
        ase::linalg::diag_matrix_product( x_inv, Z, diag_x_inv_Z );
        cblas_dgemm( ase::constants::layout, CblasNoTrans, CblasNoTrans, n, p, p,
                     1.0, diag_x_inv_Z.data( ), n, V.data( ), p, 0.0, S.data( ), n );

        // Compute lmi_inv = -S S^{T}.
        cblas_dsyrk( ase::constants::layout, ase::constants::uplo, CblasNoTrans,
                     n, p, -1.0, S.data( ), n, 0.0, lmi_inv.data( ), n );

        // Add Diag^{-1}(x) to the lmi_inv result and assign the gradient.
        for( int i = 0; i < n; i++ )
        {
          lmi_inv[ i+i*n ] += x_inv[ i ];
          grad_f_barrier_obj_at_x[ i ] = barrier_parameter*c[ i ]-lmi_inv[ i+i*n ];
        }

        // Set the call flag.
        call_f_C = false;
      };

      // Define a lambda for the barrier descent direction function.
      std::function< void( const std::vector< double >& x,
                           const std::vector< double >& grad_f_barrier_obj_at_x,
                           std::vector< double >& dx )> barrier_desc_dir;
      if( !use_alt_newton_step )
      {
        barrier_desc_dir = [ &lmi_inv, &hess, &n ]( const std::vector< double >& x,
                                                    const std::vector< double >& grad_f_barrier_obj_at_x,
                                                    std::vector< double >& dx ) -> void
        {
          /* Compute the Hessian (abs2 of each element -- F_inv is Hermitian so only
          assign the upper triangular part). Negate the Hessian instead of the gradient. */
          for( int i = 0; i < n; i++ )
            for( int j = 0; j < i+1; j++ )
              hess[ i*n+j ] = pow( lmi_inv[ i*n+j ], 2.0 );

          // Compute the descent direction.
          dx = grad_f_barrier_obj_at_x;
          cblas_dscal( n, -1.0, dx.data( ), 1 );
          LAPACKE_dposv( ase::constants::layout, ase::constants::uplo_char, n, 1,
                         hess.data( ), n, dx.data( ), n );
        };
      }
      else
      {
        barrier_desc_dir = [ &S, &lmi_inv, &hess, &n,
                             &p, &p2, &np2, &p4 ]( const std::vector< double >& x,
                                                   const std::vector< double >& grad_f_barrier_obj_at_x,
                                                   std::vector< double >& dx ) -> void
        {
          std::vector< double > b( n ), b_inv_sqrt( n );
          for( int i = 0; i < n; i++ )
          {
            /* Diagonal elements of the leading diagonal matrix of the Hessian. NOTE:
            these values will always be positive, since the Hessian is positive definite
            for a strictly feasible point (which is what we always have with an interior
            point method). */
            b[ i ] = 1.0/pow( x[ i ], 2.0 )-
                     2.0*pow( cblas_dnrm2( p, &S[ i ], n ), 2.0 )/x[ i ];
            b_inv_sqrt[ i ] = sqrt( 1.0/b[ i ]);
          }

          /* Compute T = eye( p^2 ) and R = Diag^{-1/2}(b) R0, where
          R0 = [ Diag( s_{0}) S, ... Diag( s_{p-1}) S ]. */
          int col_idx;
          std::vector< double > R( np2 ), T( p4 );
          for( int i = 0; i < p; i++ )
          {
            /* Elements to element-wise multiply by to form the low-rank term
            R = Diag^{-1/2}(b) R0, where R0 = [ Diag( s_{0}) S, ... Diag( s_{p-1}) S ]. */
            vdMul( n, b_inv_sqrt.data( ), &S[ i*n ], dx.data( ));

            for( int j = 0; j < p; j++ )
            {
              col_idx = i*p+j;
              vdMul( n, &S[ j*n ], dx.data( ), &R[ col_idx*n ]);
              T[ col_idx*p2+col_idx ] = 1.0;
            }
          }

          // Compute T := T+R^{T}R = eye( p^2 )+R^{T} R.
          cblas_dsyrk( ase::constants::layout, ase::constants::uplo, CblasTrans,
                       p2, n, 1.0, R.data( ), n, 1.0, T.data( ), p2 );

          // Compute eigendecomposition of T.
          std::vector< double > w( p2 ), V( p4 );
          MKL_INT m;
          std::vector< MKL_INT > isuppz( 2*p2 );
          LAPACKE_dsyevr( ase::constants::layout, 'V', 'A', ase::constants::uplo_char,
                          p2, T.data( ), p2, 0.0, 0.0, 0.0, 0.0, LAPACKE_dlamch( 'S' ),
                          &m, w.data( ), V.data( ), p2, isuppz.data( ));

          // Compute V := V Diag^{-1/2}(w) and assign to T.
          std::vector< double > w_inv_sqrt( p2 );
          vdInvSqrt( p2, w.data( ), w_inv_sqrt.data( ));
          T.assign( p4, 0.0 );
          ase::linalg::diag_matrix_product( w_inv_sqrt, V, T, 1.0, false );

          // Compute Diag^{-1/2}(b) R = Diag^{-1}(b) R0.
          std::vector< double > diag_b_inv_R( np2 );
          ase::linalg::diag_matrix_product( b_inv_sqrt, R, diag_b_inv_R );

          // Multiply diag_b_inv_R and T (overwrite R with the result).
          cblas_dgemm( ase::constants::layout, CblasNoTrans, CblasNoTrans, n, p2, p2,
                       1.0, diag_b_inv_R.data( ), n, T.data( ), p2, 0.0, R.data( ), n );

          /* Low-rank update for the inverse Hessian (which will be stored in
          hess) which currently contains Diag^{-1}(b). */
          cblas_dsyrk( ase::constants::layout, ase::constants::uplo, CblasNoTrans,
                       n, p2, -1.0, R.data( ), n, 0.0, hess.data( ), n );
          for( int i = 0; i < n; i++ )
            hess[ i+n*i ] += 1.0/b[ i ];

          // Compute the descent direction.
          cblas_dsymv( ase::constants::layout, ase::constants::uplo, n, -1.0,
                       hess.data( ), n, grad_f_barrier_obj_at_x.data( ), 1, 0.0,
                       dx.data( ), 1 );
        };
      }

      // Define a lambda for the barrier objective function.
      auto f_barrier_obj = [ &c, &Z, &barrier_parameter, &V, &neg_log_det_C, &call_f_C,
                             &f_C, &n ]( const std::vector< double >& x ) -> double
      {
        // Call f_C if necessary.
        if( call_f_C )
          f_C( x, Z, V, neg_log_det_C );

        // Set the call flag.
        call_f_C = true;

        // Compute the objective function.
        return barrier_parameter*cblas_ddot( n, x.data( ), 1, c.data( ), 1 )+
               ase::cvx::log_barrier( x )+neg_log_det_C;
      };

      // Apply general descent method (we can settle for a very crude solution here).
      return ase::cvx::general_descent_method_with_btls( f_barrier_obj, grad_f_barrier_obj,
                                                         barrier_desc_dir, x, 30, precision );
    };

    // Call general barrier method.
    ase::cvx::general_barrier_method( f_center, x, 1.0, 2.0, 100, precision );

    // Scale the solution x by Z_fro_norm^2.
    double Z_fro_norm2 = pow( Z_fro_norm, 2.0 );
    cblas_dscal( np, Z_fro_norm, Z.data( ), 1 );
    cblas_dscal( n2, 1.0/Z_fro_norm2, lmi_inv.data( ), 1 );
    cblas_dscal( n, Z_fro_norm2, x.data( ), 1 );
  }

  void sdp_inequality_form_with_diag_plus_low_rank_lmi( std::vector< double >& x,
                                                        const std::vector< double >& c,
                                                        std::vector< std::complex< double > >& Z,
                                                        std::vector< std::complex< double > >& lmi_inv,
                                                        const double& precision )
  {
    // Input dimension.
    int n = x.size( ), p = Z.size( )/n, n2 = n*n, p2 = p*p, np = n*p, np2 = n*p2, p4 = p2*p2;
    double neg_log_det_C;
    bool use_alt_newton_step = p2 < 0.2*n;
    std::vector< double > hess( n2 );
    std::vector< std::complex< double > > hess_cplx( n2 ), V( p2 ), S( np );

    // Initialize a strictly feasible point x.
    double Z_fro_norm = LAPACKE_zlange( ase::constants::layout, 'F', n, p, Z.data( ), n );
    cblas_zdscal( np, 1.0/Z_fro_norm, Z.data( ), 1 );
    x.assign( n, -( 1.0+pow( 10.0, -6.0 )));

    /* Define a lambda for computing the eigendecomposition of
    C = eye( p )+Z^{H} Diag^{-1}(x) Z, so that V := V Diag^{-1/2}(w) where w
    contains the eigenvalues of C and V are the eigenvectors of C. */
    auto f_C = [ &n, &p, &p2 ]( const std::vector< double >& x,
                                const std::vector< std::complex< double > >& Z,
                                std::vector< std::complex< double > >& V,
                                double& neg_log_det_C ) -> double
    {
      // Initialize C with the identity matrix.
      std::vector< std::complex< double > > C( p2 );
      for( int i = 0; i < p; i++ )
        C[ i+i*p ] = 1.0;

      // Add the term Z^{T} Diag^{-1}(x) Z.
      for( int i = 0; i < n; i++ )
        cblas_zher( ase::constants::layout, ase::constants::uplo,
                    p, 1.0/x[ i ], &Z[ i ], n, C.data( ), p );

      // Compute eigendecomposition of C = V W V^{T}.
      std::vector< double > w( p );
      MKL_INT m;
      std::vector< MKL_INT > isuppz( 2*p );
      LAPACKE_zheevr( ase::constants::layout, 'V', 'A', ase::constants::uplo_char,
                      p, C.data( ), p, 0.0, 0.0, 0.0, 0.0, LAPACKE_dlamch( 'S' ),
                      &m, w.data( ), V.data( ), p, isuppz.data( ));

      // Compute V := V Diag^{-1/2}(w)
      std::vector< double > w_inv_sqrt( p ), zeros( p );
      vdInvSqrt( p, w.data( ), w_inv_sqrt.data( ));
      std::vector< std::complex< double > > w_inv_sqrt_cplx( p );
      ase::util::complex_vector( w_inv_sqrt, zeros, w_inv_sqrt_cplx );
      C.assign( p2, 0.0 );
      ase::linalg::diag_matrix_product( w_inv_sqrt_cplx, V, C, 1.0, false );
      V = C;

      // Compute the negative logarithm of the determinant of C.
      neg_log_det_C = 0.0;
      for( int i = 0; i < p; i++ )
      {
        if( w[ i ] > 0.0 )
          neg_log_det_C -= log( w[ i ]);
        else
        {
          neg_log_det_C = std::numeric_limits< double >::infinity( );
          break;
        }
      }
      return neg_log_det_C;
    };

    // Define a lambda for the f_center function.
    auto f_center = [ &c, &Z, &f_C, &V, &neg_log_det_C, &S, &use_alt_newton_step,
                      &lmi_inv, &hess, &hess_cplx, &n, &p, &np, &n2, &p2, &np2, &p4 ]( std::vector< double >& x,
                                                                                       const double& barrier_parameter ) -> bool
    {
      // Compute the eigendecomposition of C = eye( p )+Z^{H} Diag^{-1}(x) Z.
      f_C( x, Z, V, neg_log_det_C );
      bool call_f_C;

      // Define a lambda for the gradient of the barrier objective function.
      auto grad_f_barrier_obj = [ &c, &Z, &barrier_parameter, &V, &call_f_C, &S,
                                  &lmi_inv, &n, &p, &np ]( const std::vector< double >& x,
                                                           std::vector< double >& grad_f_barrier_obj_at_x ) -> void
      {
        // Compute S = Diag^{-1}(x) Z V, where V := V Diag^{-1/2}(w).
        std::complex< double > one_cplx = { 1.0, 0.0 }, zero_cplx = { 0.0, 0.0 };
        std::vector< double > x_inv( n ), zeros( n );
        vdInv( n, x.data( ), x_inv.data( ));
        std::vector< std::complex< double > > x_inv_cplx( n ), diag_x_inv_Z( np );
        ase::util::complex_vector( x_inv, zeros, x_inv_cplx );
        ase::linalg::diag_matrix_product( x_inv_cplx, Z, diag_x_inv_Z );
        cblas_zgemm( ase::constants::layout, CblasNoTrans, CblasNoTrans, n, p, p,
                     &one_cplx, diag_x_inv_Z.data( ), n, V.data( ), p, &zero_cplx, S.data( ), n );

        // Compute lmi_inv = -S S^{T}.
        cblas_zherk( ase::constants::layout, ase::constants::uplo, CblasNoTrans,
                     n, p, -1.0, S.data( ), n, 0.0, lmi_inv.data( ), n );

        // Add Diag^{-1}(x) to the lmi_inv result and assign the gradient.
        for( int i = 0; i < n; i++ )
        {
          lmi_inv[ i+i*n ] += x_inv[ i ];
          grad_f_barrier_obj_at_x[ i ] = barrier_parameter*c[ i ]-real( lmi_inv[ i+i*n ]);
        }

        // Set the call flag.
        call_f_C = false;
      };

      // Define a lambda for the barrier descent direction function.
      std::function< void( const std::vector< double >& x,
                           const std::vector< double >& grad_f_barrier_obj_at_x,
                           std::vector< double >& dx )> barrier_desc_dir;
      if( !use_alt_newton_step )
      {
        barrier_desc_dir = [ &lmi_inv, &hess, &n ]( const std::vector< double >& x,
                                                    const std::vector< double >& grad_f_barrier_obj_at_x,
                                                    std::vector< double >& dx ) -> void
        {
          /* Compute the Hessian (abs2 of each element -- F_inv is Hermitian so only
          assign the upper triangular part). Negate the Hessian instead of the gradient. */
          for( int i = 0; i < n; i++ )
            for( int j = 0; j < i+1; j++ )
              hess[ i*n+j ] = pow( abs( lmi_inv[ i*n+j ]), 2.0 );

          // Compute the descent direction.
          dx = grad_f_barrier_obj_at_x;
          cblas_dscal( n, -1.0, dx.data( ), 1 );
          LAPACKE_dposv( ase::constants::layout, ase::constants::uplo_char, n, 1,
                         hess.data( ), n, dx.data( ), n );
        };
      }
      else
      {
        barrier_desc_dir = [ &S, &lmi_inv, &hess_cplx, &n,
                             &p, &p2, &np2, &p4 ]( const std::vector< double >& x,
                                                   const std::vector< double >& grad_f_barrier_obj_at_x,
                                                   std::vector< double >& dx ) -> void
        {
          std::vector< double > b( n );
          std::vector< std::complex< double > > b_inv_sqrt( n );
          for( int i = 0; i < n; i++ )
          {
            /* Diagonal elements of the leading diagonal matrix of the Hessian. NOTE:
            these values will always be positive, since the Hessian is positive definite
            for a strictly feasible point (which is what we always have with an interior
            point method). */
            b[ i ] = 1.0/pow( x[ i ], 2.0 )-
                     2.0*pow( cblas_dznrm2( p, &S[ i ], n ), 2.0 )/x[ i ];
            b_inv_sqrt[ i ] = sqrt( 1.0/b[ i ]);
          }

          /* Compute T = eye( p^2 ) and R = Diag^{-1/2}(b) R0, where
          R0 = [ Diag( s^{*}_{0}) S, ... Diag( s^{*}_{p-1}) S ]. */
          int col_idx;
          std::vector< std::complex< double > > R( np2 ), T( p4 ), s_i( n );
          for( int i = 0; i < p; i++ )
          {
            /* Elements to element-wise multiply by to form the low-rank term
            R = Diag^{-1/2}(b) R0, where R0 = [ Diag( s^{*}_{0}) S, ... Diag( s^{*}_{p-1}) S ]. */
            vzMul( n, b_inv_sqrt.data( ), &S[ i*n ], s_i.data( ));

            for( int j = 0; j < p; j++ )
            {
              col_idx = i*p+j;
              vzMulByConj( n, &S[ j*n ], s_i.data( ), &R[ col_idx*n ]);
              T[ col_idx*p2+col_idx ] = 1.0;
            }
          }

          // Compute T := T+R^{H}R = eye( p^2 )+R^{H} R.
          cblas_zherk( ase::constants::layout, ase::constants::uplo, CblasConjTrans,
                       p2, n, 1.0, R.data( ), n, 1.0, T.data( ), p2 );

          // Compute eigendecomposition of T.
          std::vector< double > w( p2 );
          std::vector< std::complex< double > > V( p4 );
          MKL_INT m;
          std::vector< MKL_INT > isuppz( 2*p2 );
          LAPACKE_zheevr( ase::constants::layout, 'V', 'A', ase::constants::uplo_char,
                          p2, T.data( ), p2, 0.0, 0.0, 0.0, 0.0, LAPACKE_dlamch( 'S' ),
                          &m, w.data( ), V.data( ), p2, isuppz.data( ));

          // Compute V := V Diag^{-1/2}(w) and assign to T.
          std::vector< double > w_inv_sqrt( p2 ), zeros( p2 );
          vdInvSqrt( p2, w.data( ), w_inv_sqrt.data( ));
          std::vector< std::complex< double > > w_inv_sqrt_cplx( p2 );
          ase::util::complex_vector( w_inv_sqrt, zeros, w_inv_sqrt_cplx );
          T.assign( p4, 0.0 );
          ase::linalg::diag_matrix_product( w_inv_sqrt_cplx, V, T, 1.0, false );

          // Compute Diag^{-1/2}(b) R = Diag^{-1}(b) R0.
          std::vector< std::complex< double > > diag_b_inv_R( np2 );
          ase::linalg::diag_matrix_product( b_inv_sqrt, R, diag_b_inv_R );

          // Multiply diag_b_inv_R and T (overwrite R with the result).
          std::complex< double > one_cplx = { 1.0, 0.0 }, zero_cplx = { 0.0, 0.0 };
          cblas_zgemm( ase::constants::layout, CblasNoTrans, CblasNoTrans, n, p2, p2,
                       &one_cplx, diag_b_inv_R.data( ), n, T.data( ), p2, &zero_cplx, R.data( ), n );

          /* Low-rank update for the inverse Hessian (which will be stored in
          hess) which currently contains Diag^{-1}(b). */
          cblas_zherk( ase::constants::layout, ase::constants::uplo, CblasNoTrans,
                       n, p2, -1.0, R.data( ), n, 0.0, hess_cplx.data( ), n );
          for( int i = 0; i < n; i++ )
            hess_cplx[ i+n*i ] += 1.0/b[ i ];

          // Compute the descent direction.
          std::complex< double > neg_one_cplx = { -1.0, 0.0 };
          std::vector< std::complex< double > > grad_f_barrier_obj_at_x_cplx( n ), dx_cplx( n );
          ase::util::complex_vector( grad_f_barrier_obj_at_x, zeros, grad_f_barrier_obj_at_x_cplx );
          cblas_zhemv( ase::constants::layout, ase::constants::uplo, n, &neg_one_cplx,
                       hess_cplx.data( ), n, grad_f_barrier_obj_at_x_cplx.data( ), 1, &zero_cplx,
                       dx_cplx.data( ), 1 );
          ase::util::real_vector( dx_cplx, dx );
        };
      }

      // Define a lambda for the barrier objective function.
      auto f_barrier_obj = [ &c, &Z, &barrier_parameter, &V, &neg_log_det_C, &call_f_C,
                             &f_C, &n ]( const std::vector< double >& x ) -> double
      {
        // Call f_C if necessary.
        if( call_f_C )
          f_C( x, Z, V, neg_log_det_C );

        // Set the call flag.
        call_f_C = true;

        // Compute the objective function.
        return barrier_parameter*cblas_ddot( n, x.data( ), 1, c.data( ), 1 )+
               ase::cvx::log_barrier( x )+neg_log_det_C;
      };

      // Apply general descent method.
      return ase::cvx::general_descent_method_with_btls( f_barrier_obj, grad_f_barrier_obj,
                                                         barrier_desc_dir, x, 30, pow( 10, -6.0 ));
    };

    // Call general barrier method.
    ase::cvx::general_barrier_method( f_center, x, 1.0, 2.0, 30, pow( 10.0, -6.0 ));

    // Scale the solution x by Z_fro_norm^2.
    cblas_zdscal( n*p, Z_fro_norm, Z.data( ), 1 );
    cblas_dscal( n, pow( Z_fro_norm, 2.0 ), x.data( ), 1 );
  }
} // namespace cvx
} // namespace ase
