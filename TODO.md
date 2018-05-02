#TO DO:

## Coding
1. Implement a generic add_executable function or generate a library (not sure
  which way is the best approach) call in the CMakeLists file.
1. Find a way to add_executable for all test files also.
1. Implement generic backtracking line search function. Document the function
and ensure that the documentation looks good. Implement a test file for the
backtracking line search. For the test case use a simple quadratic for the
purely real case with an integer argument specifying the dimension. Then, a test
case for  the complex case where the objective function is the norm squared of a
complex  vector.
1. Port over C++ functionality for solving SDR problem.
1. Implement a test routine for the SDR functionality.
1. Port over C++ functionality for ATSC equalization.
1. Implement a test routine for the ATSC equalization functionality.

## Documentation
1. Ensure ported over functionality is appropriately documented (may need to modify Doxyfile.in).
