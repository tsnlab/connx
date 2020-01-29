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
CONNX also depends on libprotobuf-c to parse ONNX's format

sudo apt install libprotobuf-c-dev

# Compile
make			# for debug
make RELEASE=1	# for release

# Test
make test

# License
 * CONNX is licensed under dual license GPLv3 or MIT
 * ONNX is licensed under MIT
 * re2c is licensed under public domain
 * protobuf-c is licensed under BSD-2-Clause
