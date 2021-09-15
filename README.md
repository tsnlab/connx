# Notice
We are hiring paid employee. Please contact us: contact at tsnlab dot com

# CONNX
C implementation of Open Neural Network Exchange Runtime

# Requirements
 * python3         # to run examples and test cases
 * cmake >= 3.1
 * ninja-build

# Quick start
## Compile in release mode
~~~sh
ports/linux$ mkdir build              # Make build directory
ports/linux/build$ cmake .. -G Ninja  # Generate build files with "Release" mode
ports/linux/build$ ninja              # Compile
~~~

You can find 'connx' executable in ports/linux/build directory.

## Compile with sub-opset (optional)
If you want to compile CONNX with subset of operators, in case of inferencing MNIST only, 
just make ports/linux/opset.txt file as below. And just follow the compile process.

~~~
Add Conv MatMul MaxPool Relu Reshape
~~~

## Run examples
Run MNIST example. (Mobilenet and YOLO will be coming soon)

~~~sh
ports/linux/build$ ninja mnist
~~~

Notice: If you want to run on Raspberry Pi 3, please compile with Release mode(CMAKE\_BUILD\_TYPE=Release) for sanitizer makes some problem.

# ONNX compatibility test
Run the test cases in 'test' directory which contains converted test cases from ONNX.
python3 with Numpy is required.

~~~sh
ports/linux/build$ ninja onnx
~~~

# Performance profile report
If you want to profile performance, you need to compile CONNX in debugging mode first.

~~~sh
ports/linux/build$ cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug  # Generate build files
ports/linux/build$ ninja                                       # Compile
ports/linux/build$ ninja mnist                                 # Run an any example
ports/linux/build$ ninja prof                                  # Print performance profile report
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
If you need other license than GPLv3 for proprietary use or professional support, please mail us to contact at tsnlab dot com.
