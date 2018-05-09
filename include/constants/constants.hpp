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
  const CBLAS_LAYOUT layout( CblasColMajor );
  const CBLAS_UPLO uplo( CblasUpper );
} // namespace constants
} // namespace ase
#endif // CONSTANTS_H
