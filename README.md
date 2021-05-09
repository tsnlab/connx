# C-ONNX
C implementation of Open Neural Network Exchange Runtime

# Compile
 * make # debugging
 * make RELEASE=1 # release

# Tensor I/O protoco
## To connx
input_count: int32 - -1 means quit the engine
for each input
    dtype: uint32
    ndim: uint32
    shape: uint32[] - array of uint32 values
    data: various - binary data dump

## From connx
output_count: uint32
for each output
    dtype: uint32
    ndim: uint32
    shape: uint32[] - array of uint32 values
    data: various - binary data dump

# Test
libcmocka-dev package is required.

 * make run # run all test cases

# License
 * CONNX is licensed under dual license GPLv3
