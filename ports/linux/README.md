# CONNX CLI
Usage: connx [connx model path] [input]* [output]?
       input  - tensor file, fifo or '-' for stdin(without ' mark)
                input can be omitted only when output is omitted
       output - tensor file, fifo or '-' for stdout(without ' mark)
                if output is omitted, tensor will be dump to text

## Input files, output to console
```sh
$ ./connx [connx model path] input_1.data input_2.data
                             ^- input #1  ^- input #2
$ ./connx [connx model path] input_1.data input_2.data -
                             ^- input #1  ^- input #2  ^- output
```

## Input from FIFO, output to console
```sh
$ mkfifo fifo
$ ./connx [connx model path] fifo
                             ^- input
$ ./connx [connx model path] fifo -
                        input-^   ^- output
```

## Input from FIFO, output to FIFO
```sh
$ mkfifo input_fifo
$ mkfifo output_fifo
$ ./connx [connx model path] input_fifo output_fifo
                            input-^           ^- output
```


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

