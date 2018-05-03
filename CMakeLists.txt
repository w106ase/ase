# Require current version of cmake or later.
cmake_minimum_required( VERSION "${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" )

# Define the top level project name.
project( ase )

# Use C++11.
set( CMAKE_CXX_STANDARD 11 )

# Add local cmake modules to the default cmake module path.
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" )

# Bring the header files into the project.
include_directories( include )

# Set appropriate output directories (for libraries and executables).
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build )

# Check for Intel MKL.
find_package( MKL REQUIRED )
include_directories( ${MKL_INCLUDE_DIR} )

# Gather all header files (for making a library).
set( HDR "${CMAKE_SOURCE_DIR}/include" )
file( GLOB HDR_CVXOPT "${HDR}/convexoptimization/*.*pp" )
set( ASE_HDR ${HDR_CVXOPT} )

# Gather all source files (for making a library).
set( SRC "${CMAKE_SOURCE_DIR}/src" )
file( GLOB SRC_CVXOPT "${SRC}/convexoptimization/*.cpp" )
set( ASE_SRC ${SRC_CVXOPT} )

# Build a library from the header and implementation files.
add_library( ase ${ASE_HDR} ${ASE_SRC} )

# Get all test sources.
file( GLOB_RECURSE TEST_SRC "${CMAKE_SOURCE_DIR}/test/*.cpp" )
foreach( testfile ${TEST_SRC} )
  get_filename_component( testpath ${testfile} DIRECTORY )
  get_filename_component( testnameext ${testfile} NAME )
  string( REPLACE ".cpp" "" testname ${testnameext} )
  add_executable( ${testname} "${testpath}/${testnameext}" )
  target_link_libraries( ${testname} ase ${MKL_LIBRARIES} )
endforeach( testfile ${TEST_SRC} )

# Check if Doxygen is installed.
find_package( Doxygen )
if( DOXYGEN_FOUND )
  configure_file( ${CMAKE_SOURCE_DIR}/doc/Doxyfile.in
                  ${CMAKE_SOURCE_DIR}/doc/Doxyfile.out @ONLY )

  # Add custom target to the make file.
  add_custom_target( doc ALL
                     ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/doc/Doxyfile.out
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc
                     COMMENT "Generating API documentation with Doxygen"
                     VERBATIM )
else( DOXYGEN_FOUND )
  message( STATUS "Doxygen is missing -- the documentation will not be built." )
endif( DOXYGEN_FOUND )
