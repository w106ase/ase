#TO DO:

## Coding
1. Finish documenting the SDP inequality function. Also, modify the function to
enable the LMI inverse to be returned (this will be needed for SDR).
1. Implement a test which generates test data using the MATLAB engine, then uses
CVX to solve the problem (we will need to make sure we are using MATLAB 2016 for
the CVX functionality to work).
1. Generate a complex-valued test for the general barrier method.
1. Generate a test for the log barrier function.
1. Implement a general barrier method function for complex-valued inputs.
1. Get MATLAB engine working to develop and test algorithms all in C++
1. Implement a test routine for the SDR functionality (use MATLAB engine for
visualizing and verifying functionality).
1. Implement our SDR barrier method. Use anonymous namespace to implement needed helper functions.
1. Port over C++ functionality for ATSC equalization.
1. Implement a test routine for the ATSC equalization functionality (may need to save off data set from MATLAB).
1. Port over C++ functionality for Reed-Solomon encoding/decoding.
