# C-ONNX
C implementation of Open Neural Network Exchange Runtime

# Compile
 * make # debugging
 * make RELEASE=1 # release

# Run
 make run

# Tensor I/O protocol
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
pthon3 with numpy is required

 * make test # run all test cases

# Performance report
make perf

# License
 * CONNX is licensed under dual license GPLv3
