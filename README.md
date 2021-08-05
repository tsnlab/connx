# CONNX
C implementation of Open Neural Network Exchange Runtime

# Requirements
 * python3         # to run examples and test cases
 * cmake >= 3.1
 * ninja-build

# Quick start
## Compile in release mode
~~~sh
ports/linux$ mkdir build                                            # Make build directory
ports/linux/build$ cmake .. -G Ninja -DCMAKE\_BUILD\_TYPE=Release   # Generate build files
ports/linux/build$ ninja                                            # Compile
~~~

## Compile with sub-opset (optional)
If you want to compile CONNX with subset of operators, just make ports/linux/opset.txt file like below.
And just follow the compile process.

~~~
Add Conv MatMul MaxPool Relu Reshape
~~~

## Run examples
Run MNIST example. (Mobilenet and YOLO will come soon)

~~~sh
ports/linux/build$ ninja mnist
~~~

Notice: If you want to run on Raspberry Pi 3, please compile with Release mode(CMAKE\_BUILD\_TYPE=Release) for sanitizer makes some problem.

# Test
Run the test cases in 'test' directory which contains converted test cases from ONNX.
python3 with Numpy is required.

~~~sh
ports/linux/build$ ninja test
~~~

# Performance profile report
If you want to profile performance, you need to compile CONNX in debugging mode first.

~~~sh
ports/linux/build$ cmake .. -G Ninja -DCMAKE\_BUILD\_TYPE=Debug     # Generate build files
ports/linux/build$ ninja                                            # Compile
ports/linux/build$ ninja mnist                                      # Run an any example
ports/linux/build$ ninja prof                                       # Print performance profile report
~~~

# Ports
 * See [Linux](ports/linux/README.md)
 * See [ESP32](ports/esp32/README.md)

# Contribution
See [CONTRIBUTING.md](CONTRIBUTING.md)

# Supported platforms
 * x86\_64
 * x86 - with CLFAGS=-m32
 * Raspberry pi 4 (w/ 64-bit O/S)
 * Raspberry pi 3 (32-bit O/S)

# License
CONNX is licensed under GPLv3. See [LICENSE](LICENSE)
