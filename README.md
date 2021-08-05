# CONNX
C implementation of Open Neural Network Exchange Runtime

# Requirements
 * python3         # to run examples and test cases
 * cmake >= 3.1
 * ninja-build

# Compile
~~~sh
ports/linux$ mkdir build                                            # Make build directory
ports/linux/build$ cmake .. -G Ninja -DCMAKE\_BUILD\_TYPE=Release   # Generate build files
ports/linux/build$ ninja                                            # Compile
~~~

# Compile with sub-opset
If you want to compile CONNX with subset of operators, just make ports/linux/opset.txt file like below.
And just follow the compile process.

~~~
Add Conv MatMul MaxPool Relu Reshape
~~~

# Run examples
If you want to run on Raspberry Pi 3, please compile with DEBUG=0 for to run sanitizer, some trick must be used.

~~~sh
ports/linux/build$ ninja mnist
~~~

# CONNX linux Tensor I/O protocol
CONNX linux port reads and writes tensor via Linux pipe. Below is the short description of tensor I/O protocol.

## To CONNX
input\_count: int32 - -1 means terminate the engine

for each input  

 * dtype: uint32
 * ndim: uint32
 * shape: uint32[] - array of uint32 values
 * data: various - binary data dump

## From CONNX
output\_count: uint32

for each output

 * dtype: uint32
 * ndim: uint32
 * shape: uint32[] - array of uint32 values
 * data: various - binary data dump

# How to add new operator
 1. Implement operator in src/opset directory
 2. Convert ONNX test case to CONNX using onnx-connx's bin/convert utility
 3. ports/linux$ ninja test

# Test
Run the test cases in 'test' directory which contains converted the test cases from ONNX.
pthon3 with Numpy is required.

~~~sh
ports/linux/build$ ninja test
~~~

# Performance profile report
If you want to profile performance, you need to compile CONNX in debugging mode.

~~~sh
ports/linux/build$ cmake .. -G Ninja -DCMAKE\_BUILD\_TYPE=Debug     # Generate build files
ports/linux/build$ ninja                                            # Compile
ports/linux/build$ ninja mnist                                      # Run an any example
ports/linux/build$ ninja prof                                       # Print performance profile report
~~~

# Supported platforms
 * x86\_64
 * x86 - with CLFAGS=-m32
 * Raspberry pi 4 (w/ 64-bit O/S)
 * Raspberry pi 3 (32-bit O/S)

# License
 * CONNX is licensed under GPLv3
