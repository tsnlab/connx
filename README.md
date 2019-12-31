# CONNX
C implementation of Open Neural Network Exchange Runtime

# Install
## ONNX submodule
CONNX depends on ONNX

sudo apt install git
git submodule init
git submodule update

## libc
CONNX is based on C language

sudo apt install libc6-dev

## Protocol buffer
CONNX also depends on libprotobuf-c to parse ONNX's format

sudo apt install libprotobuf-c-dev

# Compile
make			# for debug
make release	# for release

# Test
make test

# License
 * CONNX can be distributed under dual license, GPL3 or MIT.
 * ONNX submodule is MIT license
 * re2c is public domain license
 * protobuf-c is BSD-2-Clause license
