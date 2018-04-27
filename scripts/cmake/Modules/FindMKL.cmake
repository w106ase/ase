# Find the MKL libraries
#
# This module defines the following variables:
#
#   MKL_FOUND            : true if mkl is found
#   MKL_INCLUDE_DIR      : include directory
#   MKL_LIBRARIES        : the libraries to link against.


# Set expected location of MKL directories on macOS, Windows and Unix/Linux
if( APPLE )
  set( INTEL_DIR "/opt/intel" )
  set( MKL_INCLUDE_DIR ${INTEL_DIR}/mkl/include )
  set( MKL_LIB_DIR ${INTEL_DIR}/mkl/lib )
  set( INTEL_LIB_DIR ${INTEL_DIR}/lib )
elseif( WIN32 )
  set( INTEL_DIR "C:/Program Files\ (x86)/IntelSWTools/compilers_and_libraries/windows" CACHE PATH "Path to Intel folder" FORCE)
  set( MKL_INCLUDE_DIR ${INTEL_DIR}/mkl/include )
  set( MKL_LIB_DIR ${INTEL_DIR}/mkl/lib/intel64 )
  set( INTEL_LIB_DIR ${INTEL_DIR}/compiler/lib/intel64 )
else() #( UNIX )
  set( INTEL_DIR "/opt/intel" )
  set( MKL_INCLUDE_DIR ${INTEL_DIR}/mkl/include )
  set( MKL_LIB_DIR ${INTEL_DIR}/mkl/lib/intel64 )
  set( INTEL_LIB_DIR ${INTEL_DIR}/lib/intel64 )
endif()
#message( STATUS "MKL_INCLUDE_DIR: ${MKL_INCLUDE_DIR}")
#message( STATUS "MKL_LIB_DIR: ${MKL_LIB_DIR}")
# Find MKL include directory
#find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKL_DIR} PATH_SUFFIXES include)

# Set prefix and suffix for libraries
if(APPLE)
  set( MKL_LIB_PREFIX lib )
  set( MKL_LIB_SUFFIX .a )
elseif( WIN32 )
  set( MKL_LIB_SUFFIX .lib )
else() # Linux
  set( MKL_LIB_PREFIX lib )
  set( MKL_LIB_SUFFIX .a )
endif()

# Find interface layer library
find_library( MKL_INTERFACE_LIBRARY ${MKL_LIB_PREFIX}mkl_intel_ilp64${MKL_LIB_SUFFIX}
  PATHS ${MKL_LIB_DIR}
  )

# Find threading layer library
find_library( MKL_THREADING_LIBRARY ${MKL_LIB_PREFIX}mkl_intel_thread${MKL_LIB_SUFFIX}
  PATHS ${MKL_LIB_DIR}
  )

# Find computational layer library
find_library( MKL_CORE_LIBRARY ${MKL_LIB_PREFIX}mkl_core${MKL_LIB_SUFFIX}
  PATHS ${MKL_LIB_DIR}
  )

# Find run-time library (RTL)
#find_library( MKL_RTL_LIBRARY ${MKL_LIB_PREFIX}iomp5${MKL_LIB_SUFFIX}
if( WIN32 )
  find_library( MKL_RTL_LIBRARY libiomp5md
    PATHS ${INTEL_LIB_DIR}
    )
  find_file( MKL_RTL_LIBRARY_DLL libiomp5md.dll
    PATHS ${INTEL_DIR}/redist/intel64/compiler
	)
  #message( STATUS "INTEL_LIB_DIR: ${INTEL_LIB_DIR}" )
  #message( STATUS "MKL_RTL_LIBRARY: ${MKL_RTL_LIBRARY}" )
  #message( STATUS "MKL_RTL_LIBRARY_DLL: ${MKL_RTL_LIBRARY_DLL}" )
else()
  find_library( MKL_RTL_LIBRARY iomp5
    PATHS ${INTEL_LIB_DIR}
    )
endif()

# Store MKL libraries in a variable
if( UNIX AND NOT APPLE )
  set( MKL_LIBRARY "-Wl,--start-group ${MKL_INTERFACE_LIBRARY} ${MKL_THREADING_LIBRARY} ${MKL_CORE_LIBRARY} -Wl,--end-group" )
else()
  set( MKL_LIBRARY
    ${MKL_INTERFACE_LIBRARY}
    ${MKL_THREADING_LIBRARY}
    ${MKL_CORE_LIBRARY}
    )
endif()
set( MKL_LIBRARY ${MKL_LIBRARY} ${MKL_RTL_LIBRARY} )
if( UNIX AND NOT APPLE )
  set( MKL_LIBRARY ${MKL_LIBRARY} pthread m dl )
endif()

# Handle find package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args( MKL DEFAULT_MSG MKL_INCLUDE_DIR )
if( MKL_FOUND )
  set( MKL_LIBRARIES ${MKL_LIBRARY} )
endif()

# Set path to iomp5 dll on windows
#if( WIN32 )
#  set( MKL_IOMP5_DLL ${INTEL_DIR}/../../compilers_and_libraries_2017.4.210/windows/redist/intel64/compiler/libiomp5md.dll )
#endif()
