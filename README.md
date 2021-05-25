# C-ONNX
C implementation of Open Neural Network Exchange Runtime

# Compile
 * make # debugging
 * make DEBUG=0 # release

# Run
If you want to run on Raspberry Pi 3 please compile with DEBUG=0 for to run sanitizer, some trick must be used.
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
