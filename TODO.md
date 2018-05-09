#TO DO:

## Coding
1. Implement a generalized log barrier funtion. We will need two separate
functions: 1) takes the matrix as the argument and computes the determinant
using a brute-force approach (will have a flag for negating but otherwise the
matrix should be positive definite) and 2) takes the matrix as the argument as
well as a function which computes the determinant efficiently.
1. Generate a test for the log and generalized log barrier functions.
1. Implement a general barrier method function for complex-valued inputs.
1. Finish documenting and implementing the barrier method function test (need to
add the complex-valued case).
1. Get MATLAB engine working to develop and test algorithms all in C++
1. Implement a test routine for the SDR functionality (use MATLAB engine for
visualizing and verifying functionality).
1. Implement our SDR barrier method. Use anonymous namespace to implement needed helper functions.
1. Port over C++ functionality for ATSC equalization.
1. Implement a test routine for the ATSC equalization functionality (may need to save off data set from MATLAB).
1. Port over C++ functionality for Reed-Solomon encoding/decoding.
