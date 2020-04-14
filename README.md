# CONNX
C implementation of Open Neural Network Exchange Runtime

# Install
## ONNX submodule
CONNX depends on ONNX

git submodule init
git submodule update

## libc
CONNX is based on C language

sudo apt install libc6-dev

## Protocol buffer
CONNX also depends on libprotobuf-c and protobuf-c-compiler to parse ONNX's format

sudo apt install libprotobuf-c-dev protobuf-c-compiler

# Compile
make			# for debug
make RELEASE=1	# for release

# Run examples
make example_mnist	# for MNIST example
make example_yolo	# for YOLO example

# Test
To test CONNX operators, autoreconf and libtool utility is needded to compile re2c

sudo apt install autoconf libtool

make test

# Dump onnx file
bin/dump [onnx file]

# License
 * CONNX is licensed under dual license GPLv3 or MIT
 * ONNX is licensed under MIT
 * re2c is licensed under public domain
 * protobuf-c is licensed under BSD-2-Clause
