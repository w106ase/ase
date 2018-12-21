/*! \file testDualSDPInequalityFormDiagPlusLowRank.cpp
    \brief Demonstrates the dual SDP inequality form with diagonal plus low-rank
    data functions using both real- and complex-valued data.

    Demonstrates the dual SDP inequality form with diagonal plus low-rank data
    functions using both real- and complex-valued data. For the demonstrations,
    the result is compared to that obtained using CVX with a MATLAB engine. The
    specific functions that are exercised is/are:
    1. dual_sdp_inequality_form_with_diag_plus_low_rank_lmi( )
    2. dual_sdp_inequality_form_with_diag_plus_low_rank_lmi( )
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
void complex_valued_example( Engine *&ep );

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
  engEvalString( ep, "n = 300;" );
  engEvalString( ep, "p = 20;" );
  engEvalString( ep, "c = -ones( n, 1 );" );
  engEvalString( ep, "Z = randn( n, p );" );
  engEvalString( ep, "Z = Z/norm( Z, 'fro' );" );

  // Get the problem dimension.
  int n = ( int ) mxGetScalar( engGetVariable( ep, "n" ));
  int p = ( int ) mxGetScalar( engGetVariable( ep, "p" ));

  /* Solve the problem using MATLAB CVX.
  NOTE: CVX takes roughly 10 min. or more to run for dimension n > 1000. */
  bool run_cvx = n < 1000;
  double t_cvx_s;
  if( run_cvx )
  {
    engEvalString( ep, "cvx_precision( 'low' )" );
    engEvalString( ep, "tic;" );
    engEvalString( ep, "cvx_begin quiet sdp" );
    engEvalString( ep, "variable Phi( n, n ) semidefinite" );
    engEvalString( ep, "dual variable x" );
    engEvalString( ep, "maximize( trace( Phi*( Z*Z' )))" );
    engEvalString( ep, "subject to" );
    engEvalString( ep, "x : diag( Phi ) == ones( n, 1 )" );
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
  vector< double > x( n ), x_cvx( n ), Phi( n*n ), Phi_cvx( n*n );
  if( run_cvx )
  {
  	mxArray *x_ptr = engGetVariable( ep, "x" );
  	memcpy( x_cvx.data( ), mxGetPr( x_ptr ), n*sizeof( mxGetClassName( x_ptr )));
    mxDestroyArray( x_ptr );
  	mxArray *Phi_ptr = engGetVariable( ep, "Phi" );
  	memcpy( Phi_cvx.data( ), mxGetPr( Phi_ptr ), n*sizeof( mxGetClassName( Phi_ptr )));
    mxDestroyArray( Phi_ptr );
  }

  // C++ solver.
  bool comp_chol = true;
  double t_start_s = dsecnd( );
  ase::cvx::dual_sdp_inequality_form_with_diag_plus_low_rank_lmi( c, Z, Phi, x, comp_chol, 1.0e-3 );
  double t_stop_s = dsecnd( );
  cout << "Estimated C++ function run-time " << t_stop_s-t_start_s << " sec." << endl;

  // Create mxArray to store C++ result.
  mxArray *Phi_c = mxCreateDoubleMatrix( n*n, 1, mxREAL );
  memcpy( mxGetPr( Phi_c ), Phi.data( ), n*n*sizeof( double ));
  engPutVariable( ep, "Phi_c", Phi_c );
  engEvalString( ep, "Phi_c = reshape( Phi_c, n, n );" );
  if( comp_chol )
    engEvalString( ep, "Phi_c = Phi_c'*Phi_c;" );
  else
    engEvalString( ep, "Phi_c = Phi_c+Phi_c'-eye( n );" );
  mxDestroyArray( Phi_c );
  mxArray *x_c = mxCreateDoubleMatrix( n, 1, mxREAL );
  memcpy( mxGetPr( x_c ), x.data( ), n*sizeof( double ));
  engPutVariable( ep, "x_c", x_c );
  mxDestroyArray( x_c );

  // Evaluate suboptimality and MSE.
  engEvalString( ep, "c_opt = trace( Phi_c*( Z*Z' ));" );
  if( run_cvx )
  {
    engEvalString( ep, "mse = mean( abs( Phi( : )-Phi_c( : )).^2 );" );
    cout << "MSE: " << mxGetScalar( engGetVariable( ep, "mse" )) << endl;
    cout << "CVX objective function: " << mxGetScalar( engGetVariable( ep, "cvx_optval" )) << endl;
  }
  cout << "C++ objective function: " << mxGetScalar( engGetVariable( ep, "c_opt" )) << endl;
  cout << "Primal/Dual Difference (i.e., suboptimality): " << cblas_ddot( n, x.data( ), 1, c.data( ), 1 )-mxGetScalar( engGetVariable( ep, "c_opt" )) << endl;
  cout << endl;
}

void complex_valued_example( Engine *&ep )
{
  // Generate problem instance in MATLAB.
  engEvalString( ep, "rng( 0 );" );
  engEvalString( ep, "n = 300;" );
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
  NOTE: CVX appears to have a bug when n >= 500. For instance, the CVX output
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
    engEvalString( ep, "variable Phi( n, n ) hermitian semidefinite" );
    engEvalString( ep, "dual variable x" );
    engEvalString( ep, "maximize( trace( Phi*( Z*Z' )))" );
    engEvalString( ep, "subject to" );
    engEvalString( ep, "x : diag( Phi ) == ones( n, 1 )" );
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
  vector< complex< double > > Phi( n*n ), Phi_cvx( n*n );
  if( run_cvx )
  {
  	mxArray *x_ptr = engGetVariable( ep, "x" );
  	memcpy( x_cvx.data( ), mxGetPr( x_ptr ), n*sizeof( mxGetClassName( x_ptr )));
    mxDestroyArray( x_ptr );
  	mxArray *Phi_ptr = engGetVariable( ep, "Phi" );
  	memcpy( Phi_cvx.data( ), mxGetPr( Phi_ptr ), n*sizeof( mxGetClassName( Phi_ptr )));
    mxDestroyArray( Phi_ptr );
  }

  // C++ solver.
  bool comp_chol = true;
  double t_start_s = dsecnd( );
  ase::cvx::dual_sdp_inequality_form_with_diag_plus_low_rank_lmi( c, Z, Phi, x, true, 1.0e-3 );
  double t_stop_s = dsecnd( );
  cout << "Estimated C++ function run-time " << t_stop_s-t_start_s << " sec." << endl;

  // Create mxArray to store C++ result.
  mxArray *Phi_c = mxCreateDoubleMatrix( n*n, 1, mxCOMPLEX );
  vector< double > Phi_re( n*n ), Phi_im( n*n );
  ase::util::real_vector( Phi, Phi_re );
  ase::util::imag_vector( Phi, Phi_im );
  memcpy( mxGetPr( Phi_c ), Phi_re.data( ), n*n*sizeof( double ));
  memcpy( mxGetPi( Phi_c ), Phi_im.data( ), n*n*sizeof( double ));
  engPutVariable( ep, "Phi_c", Phi_c );
  engEvalString( ep, "Phi_c = reshape( Phi_c, n, n );" );
  if( comp_chol )
    engEvalString( ep, "Phi_c = Phi_c'*Phi_c;" );
  else
    engEvalString( ep, "Phi_c = Phi_c+Phi_c'-eye( n );" );
  mxDestroyArray( Phi_c );
  mxArray *x_c = mxCreateDoubleMatrix( n, 1, mxREAL );
  memcpy( mxGetPr( x_c ), x.data( ), n*sizeof( double ));
  engPutVariable( ep, "x_c", x_c );
  mxDestroyArray( x_c );

  // Evaluate suboptimality and MSE.
  engEvalString( ep, "c_opt = trace( Phi_c*( Z*Z' ));" );
  if( run_cvx )
  {
    engEvalString( ep, "mse = mean( abs( Phi( : )-Phi_c( : )).^2 );" );
    cout << "MSE: " << mxGetScalar( engGetVariable( ep, "mse" )) << endl;
    cout << "CVX objective function: " << mxGetScalar( engGetVariable( ep, "cvx_optval" )) << endl;
  }
  cout << "C++ objective function: " << mxGetScalar( engGetVariable( ep, "c_opt" )) << endl;
  cout << "Primal/Dual Difference (i.e., suboptimality): " << cblas_ddot( n, x.data( ), 1, c.data( ), 1 )-mxGetScalar( engGetVariable( ep, "c_opt" )) << endl;
  cout << endl;
}
#endif
