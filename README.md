# C-ONNX
C implementation of Open Neural Network Exchange Runtime

# Compile
 * ports/linux$ make # debug
 * ports/linux$ make DEBUG=0 # release

# Run
If you want to run on Raspberry Pi 3, please compile with DEBUG=0 for to run sanitizer, some trick must be used.
 * ports/linux$ make run

# Tensor I/O protocol
## To connx
input\_count: int32 - -1 means quit the engine
for each input
 * dtype: uint32
 * ndim: uint32
 * shape: uint32[] - array of uint32 values
 * data: various - binary data dump

## From connx
output\_count: uint32
for each output
 * dtype: uint32
 * ndim: uint32
 * shape: uint32[] - array of uint32 values
 * data: various - binary data dump

## Add new operator
 1. Implement operator in src/opset directory
 2. Convert ONNX test case to CONNX using onnx-connx's bin/convert utility
 3. ports/linux$ make test

# Test
pthon3 with Numpy is required

 * ports/linux$ make test # run all test cases

# Performance report
 * ports/linux$ make perf

# Supported platforms
 * x86\_64
 * x86 - make CLFAGS=-m32
 * Raspberry pi 4 (w/ 64-bit O/S)
 * Raspberry pi 3 (32-bit O/S) - make DEBUG=0

# License
 * CONNX is licensed under GPLv3
