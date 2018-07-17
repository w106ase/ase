/*! \file constants.hpp
    \brief Constants used within the namespace ase.

    Constants used within the namespace ase. These include the default matrix
    layout used with Intel MKL routines, whether to store the upper or lower
    triangular part of a symmetric or Hermitian matrix, etc.
*/

#ifndef CONSTANTS_H
#define CONSTANTS_H
#include "mkl.h"

namespace ase
{
namespace constants
{
  const CBLAS_LAYOUT layout( CblasColMajor ); /*!< CBLAS layout */
  const CBLAS_UPLO uplo( CblasUpper ); /*!< CBLAS uplo */
  const char uplo_char( 'U' ); /*!< CBLAS uplo character */
  const double btls_alpha = 0.1; /*!< Backtracking line search alpha */
  const double btls_beta = 0.8; /*!< Backtracking line search alpha */
  const double btls_max_iter = 200; /*!< Backtracking line search maximum iterations */
  const double gbm_barrier_param0 = 1.0; /*!< General barrier method initial barrier parameter */
  const double gbm_barrier_param_update = 1.5; /*!< General barrier method barrier parameter update */
  const double gbm_max_iter = 100; /*!< General barrier method maximum iterations */
  const double gbm_sub_opt_thresh = 1.0e-6; /*!< General barrier method suboptimality threshold */
  const double gdm_max_iter = 100; /*!< General descent method maximum iterations */
  const double gdm_norm2_grad_thresh = 1.0e-6; /*!< General descent method norm-squared gradient threshold */
  const double gdm_diff_f_obj_thresh = 1.0e-3; /*!< General descent method difference in objective function threshold */
  const double gdm_rel_dx_norm2_thresh = 1.0e-9; /*!< General descent method relative difference threshold */
  const double low_prec = 1.0e-3; /*!< Low precision solution */
  const double med_prec = 1.0e-6; /*!< Medium precision solution */
  const double high_prec = 1.0e-9; /*!< High precision solution */
} // namespace constants
} // namespace ase
#endif // CONSTANTS_H
