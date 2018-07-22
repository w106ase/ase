/*! \file testSDPInequalityFormDiagPlusLowRank.cpp
    \brief Demonstrates the SDP inequality form with diagonal plus low rank data.
*/

#include <iostream>
#include <complex>
#include <vector>
#include "convexoptimization/cvxopt.hpp"
#include "utility/util.hpp"
#include "engine.h"
#include "mkl.h"

using namespace std;

#ifndef DOXYGEN_SKIP
void real_valued_example( Engine *&ep );
void complex_valued_example( Engine *&p );

int main( int argc, char* argv[ ])
{
  // Create MATLAB engine.
  Engine *ep;
  ep = engOpen( "" );

  // Real-valued examples.
  cout << "Example Result #1 Using Real-Valued Arguments" << endl;
  cout << "-----------------------------------------------------------" << endl;
  cout << endl;
  srand( 0 );
  real_valued_example( ep );
  cout << endl;

  // Complex-valued examples.
  cout << "Example Result #1 Using Complex-Valued Arguments" << endl;
  cout << "-----------------------------------------------------------" << endl;
  cout << endl;
  srand( 0 );
  complex_valued_example( ep );
  cout << endl;

  // Close MATLAB engine.
  engClose( ep );

  return 0;
}

void real_valued_example( Engine *&ep )
{
  // Generate problem instance in MATLAB.
  engEvalString( ep, "rng( 0 );" );
  engEvalString( ep, "n = 400;" );
  engEvalString( ep, "p = 20;" );
  engEvalString( ep, "c = -ones( n, 1 );" );
  engEvalString( ep, "Z = randn( n, p );" );
  engEvalString( ep, "Z = Z/norm( Z, 'fro' );" );

  // Get the problem dimension.
  int n = ( int ) mxGetScalar( engGetVariable( ep, "n" ));
  int p = ( int ) mxGetScalar( engGetVariable( ep, "p" ));

  /* Solve the problem using MATLAB CVX.
  NOTE: CVX appears to have a bug when n >= 1000. For instance, the CVX output
  states that dual and primal solutions are obtained which are separated by some
  specified threshold. However, evaluating the objective function for each of
  these solutions shows that the primal objective function value appears to be
  off by approximately a factor of 2. The problem becomes more exacerbated as
  the dimension n grows. */
  bool run_cvx = n < 1000;
  double t_cvx_s;
  if( run_cvx )
  {
    engEvalString( ep, "cvx_precision( 'low' )" );
    engEvalString( ep, "tic;" );
    engEvalString( ep, "cvx_begin quiet sdp" );
    engEvalString( ep, "variable x( n )" );
    engEvalString( ep, "minimize( c'*x )" );
    engEvalString( ep, "subject to" );
    engEvalString( ep, "diag( x )+Z*Z' <= 0" );
    engEvalString( ep, "cvx_end" );
    engEvalString( ep, "t_cvx_s = toc;" );
    t_cvx_s = mxGetScalar( engGetVariable( ep, "t_cvx_s" ));
    cout << "Estimated CVX run-time " << t_cvx_s << " sec." << endl;
  }

  // Get the problem data from MATLAB.
  mxArray *Z_ptr = engGetVariable( ep, "Z" ), *c_ptr = engGetVariable( ep, "c" );
  vector< double > Z( n*p ), c( n );
  memcpy( Z.data( ), mxGetPr( Z_ptr ), n*p*sizeof( mxGetClassName( Z_ptr )));
  mxDestroyArray( Z_ptr );
  memcpy( c.data( ), mxGetPr( c_ptr ), n*sizeof( mxGetClassName( c_ptr )));
  mxDestroyArray( c_ptr );

  // Get the solution from MATLAB CVX.
  vector< double > x( n ), x_cvx( n );
  if( run_cvx )
  {
  	mxArray *x_ptr = engGetVariable( ep, "x" );
  	memcpy( x_cvx.data( ), mxGetPr( x_ptr ), n*sizeof( mxGetClassName( x_ptr )));
    mxDestroyArray( x_ptr );
  }

  // C++ solver.
  vector< double > lmi_inv( n*n );
  double t_start_s = dsecnd( );
  ase::cvx::sdp_inequality_form_with_diag_plus_low_rank_lmi( x, c, Z, lmi_inv, 1.0e-3 );
  double t_stop_s = dsecnd( );
  cout << "Estimated C++ function run-time " << t_stop_s-t_start_s << " sec." << endl;
  cout << endl;

  // Create mxArray to store C++ result.
  mxArray *F_inv = mxCreateDoubleMatrix( n*n, 1, mxREAL );
  memcpy( mxGetPr( F_inv ), lmi_inv.data( ), n*n*sizeof( double ));
  engPutVariable( ep, "F_inv", F_inv );
  engEvalString( ep, "F_inv = reshape( F_inv/F_inv( 1 ), n, n );" );
  engEvalString( ep, "F_inv = F_inv+F_inv'-eye( n );" );
  mxDestroyArray( F_inv );
  mxArray *x_c = mxCreateDoubleMatrix( n, 1, mxREAL );
  memcpy( mxGetPr( x_c ), x.data( ), n*sizeof( double ));
  engPutVariable( ep, "x_c", x_c );
  mxDestroyArray( x_c );

  // Evaluate suboptimality and maximum eigenvalue.
  if( run_cvx )
  {
    engEvalString( ep, "max_eig = max( eig( diag( x )+Z*Z' ));" );
    cout << "Max. Eigenvalue CVX: " << mxGetScalar( engGetVariable( ep, "max_eig" )) << endl;
  }
  engEvalString( ep, "max_eig_c = max( eig( diag( x_c )+Z*Z' ));" );
  cout << "Max. Eigenvalue C++: " << mxGetScalar( engGetVariable( ep, "max_eig_c" )) << endl;
  engEvalString( ep, "dual = trace( F_inv*( Z*Z' ));" );
  if( run_cvx )
  {
    engEvalString( ep, "mse = mean( abs( x-x_c ).^2 );" );
    cout << "MSE: " << mxGetScalar( engGetVariable( ep, "mse" )) << endl;
    cout << "CVX objective function: " << cblas_ddot( n, x_cvx.data( ), 1, c.data( ), 1 ) << endl;
  }
  cout << "C++ objective function: " << cblas_ddot( n, x.data( ), 1, c.data( ), 1 ) << endl;
  cout << "Primal/Dual Difference (i.e., suboptimality): " << cblas_ddot( n, x.data( ), 1, c.data( ), 1 )-mxGetScalar( engGetVariable( ep, "dual" )) << endl;
  cout << endl;
}

void complex_valued_example( Engine *&ep )
{
  // Generate problem instance in MATLAB.
  engEvalString( ep, "rng( 0 );" );
  engEvalString( ep, "n = 400;" );
  engEvalString( ep, "p = 20;" );
  engEvalString( ep, "c = -ones( n, 1 );" );
  engEvalString( ep, "Z = randn( n, p )+1i*randn( n, p );" );
  engEvalString( ep, "Z = Z/norm( Z, 'fro' );" );
  engEvalString( ep, "Z_re = real( Z );" );
  engEvalString( ep, "Z_im = imag( Z );" );

  // Get the problem dimension.
  int n = ( int ) mxGetScalar( engGetVariable( ep, "n" ));
  int p = ( int ) mxGetScalar( engGetVariable( ep, "p" ));

  /* Solve the problem using MATLAB CVX.
  NOTE: CVX appears to have a bug when n >= 1000. For instance, the CVX output
  states that dual and primal solutions are obtained which are separated by some
  specified threshold. However, evaluating the objective function for each of
  these solutions shows that the primal objective function value appears to be
  off by approximately a factor of 2. The problem becomes more exacerbated as
  the dimension n grows. */
  bool run_cvx = n < 500;
  double t_cvx_s;
  if( run_cvx )
  {
    engEvalString( ep, "cvx_precision( 'low' )" );
    engEvalString( ep, "tic;" );
    engEvalString( ep, "cvx_begin quiet sdp" );
    engEvalString( ep, "variable x( n )" );
    engEvalString( ep, "minimize( c'*x )" );
    engEvalString( ep, "subject to" );
    engEvalString( ep, "diag( x )+Z*Z' <= 0" );
    engEvalString( ep, "cvx_end" );
    engEvalString( ep, "t_cvx_s = toc;" );
    t_cvx_s = mxGetScalar( engGetVariable( ep, "t_cvx_s" ));
    cout << "Estimated CVX run-time " << t_cvx_s << " sec." << endl;
  }

  // Get the problem data from MATLAB.
  mxArray *Z_re_ptr = engGetVariable( ep, "Z_re" ),
          *Z_im_ptr = engGetVariable( ep, "Z_im" ), *c_ptr = engGetVariable( ep, "c" );
  vector< double > Z_re( n*p ), Z_im( n*p ), c( n );
  vector< complex< double > > Z( n*p );
  memcpy( Z_re.data( ), mxGetPr( Z_re_ptr ), n*p*sizeof( double ));
  memcpy( Z_im.data( ), mxGetPr( Z_im_ptr ), n*p*sizeof( double ));
  mxDestroyArray( Z_re_ptr );
  mxDestroyArray( Z_im_ptr );
  ase::util::complex_vector( Z_re, Z_im, Z );
  memcpy( c.data( ), mxGetPr( c_ptr ), n*sizeof( mxGetClassName( c_ptr )));
  mxDestroyArray( c_ptr );

  // Get the solution from MATLAB CVX.
  vector< double > x( n ), x_cvx( n );
  if( run_cvx )
  {
  	mxArray *x_ptr = engGetVariable( ep, "x" );
  	memcpy( x_cvx.data( ), mxGetPr( x_ptr ), n*sizeof( mxGetClassName( x_ptr )));
    mxDestroyArray( x_ptr );
  }

  // C++ solver.
  vector< complex< double > > lmi_inv( n*n );
  double t_start_s = dsecnd( );
  ase::cvx::sdp_inequality_form_with_diag_plus_low_rank_lmi( x, c, Z, lmi_inv, 1.0e-3 );
  double t_stop_s = dsecnd( );
  cout << "Estimated C++ function run-time " << t_stop_s-t_start_s << " sec." << endl;
  cout << endl;

  // Create mxArray to store C++ result.
  mxArray *F_inv = mxCreateDoubleMatrix( n*n, 1, mxCOMPLEX );
  vector< double > lmi_inv_re( n*n ), lmi_inv_im( n*n );
  ase::util::real_vector( lmi_inv, lmi_inv_re );
  ase::util::imag_vector( lmi_inv, lmi_inv_im );
  memcpy( mxGetPr( F_inv ), lmi_inv_re.data( ), n*n*sizeof( double ));
  memcpy( mxGetPi( F_inv ), lmi_inv_im.data( ), n*n*sizeof( double ));
  engPutVariable( ep, "F_inv", F_inv );
  engEvalString( ep, "F_inv = reshape( F_inv/F_inv( 1 ), n, n );" );
  engEvalString( ep, "F_inv = F_inv+F_inv'-eye( n );" );
  mxDestroyArray( F_inv );
  mxArray *x_c = mxCreateDoubleMatrix( n, 1, mxREAL );
  memcpy( mxGetPr( x_c ), x.data( ), n*sizeof( double ));
  engPutVariable( ep, "x_c", x_c );
  mxDestroyArray( x_c );

  // Evaluate suboptimality and maximum eigenvalue.
  if( run_cvx )
  {
    engEvalString( ep, "max_eig = max( real( eig( diag( x )+Z*Z' )));" );
    cout << "Max. Eigenvalue CVX: " << mxGetScalar( engGetVariable( ep, "max_eig" )) << endl;
  }
  engEvalString( ep, "max_eig_c = max( real( eig( diag( x_c )+Z*Z' )));" );
  cout << "Max. Eigenvalue C++: " << mxGetScalar( engGetVariable( ep, "max_eig_c" )) << endl;
  engEvalString( ep, "dual = real( trace( F_inv*( Z*Z' )));" );
  if( run_cvx )
  {
    engEvalString( ep, "mse = mean( abs( x-x_c ).^2 );" );
    cout << "MSE: " << mxGetScalar( engGetVariable( ep, "mse" )) << endl;
    cout << "CVX objective function: " << cblas_ddot( n, x_cvx.data( ), 1, c.data( ), 1 ) << endl;
  }
  cout << "C++ objective function: " << cblas_ddot( n, x.data( ), 1, c.data( ), 1 ) << endl;
  cout << "Primal/Dual Difference (i.e., suboptimality): " << cblas_ddot( n, x.data( ), 1, c.data( ), 1 )-mxGetScalar( engGetVariable( ep, "dual" )) << endl;
  cout << endl;
}
#endif
