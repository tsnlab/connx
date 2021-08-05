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

