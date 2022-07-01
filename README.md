# Notice
We are hiring paid employee. Please contact us: contact at tsnlab dot com

# CONNX
C implementation of Open Neural Network Exchange Runtime

# Requirements
 * python3 >= 3.8  # To build templates and for python bindings
 * [poetry][]      # To setup python develop environment
 * cmake >= 3.1
 * ninja-build

[poetry]: https://pypi.org/project/poetry/

# Quick start

## Compile in release mode
~~~sh
connx$ poetry config --local virtualenvs.create false  # To use python directly
connx$ poetry install --no-dev --no-root  # To install python dependencies
connx$ mkdir build; cd build                # Make build directory
connx/build$ cmake ../ports/linux -G Ninja -D CMAKE_BUILD_TYPE=Release  # Generate build files with "Release" mode
connx/build$ ninja                          # Compile
~~~

You can find 'connx' executable in connx/build directory.

## Build python bindings

```sh
# Install poetry using pipx or pip first
$ poetry build
```

It will automatically compiles library and build sdist, wheel archive on `dist` directory.

## Compile with sub-opset (optional)
If you want to compile CONNX with subset of operators, in case of inferencing MNIST only,
just make ports/linux/opset.txt file as below. And just follow the compile process.

~~~
Add Conv MatMul MaxPool Relu Reshape
~~~

## Run examples
Run MNIST example. (Mobilenet and YOLO will be coming soon)

~~~sh
connx$ cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug ports/linux  # Change Debug→Release for release mode
connx$ ninja -C build mnist
~~~

Notice: If you want to run on Raspberry Pi 3, please compile with Release mode(CMAKE\_BUILD\_TYPE=Release) for sanitizer makes some problem.

## Using python bindings

```py
import connx

model = connx.load_model('examples/mnist')
input_data = connx.load_data('examples/mnist/test_data_set_0/input_0.data')
reference_data = connx.load_data('examples/mnist/test_data_set_0/output_0.data')

# Run model with input data
output_data = model.run([input_data])
# output_data is an array that contains output tensors

# Convert to numpy ndarray
reference_nparray = reference_data.to_nparray()
output_nparray = output_data[0].to_nparray

# Check output with reference_data
assert reference_data.shape == output_data[0].shape
import numpy
numpy.allclose(reference_nparray, output_nparray)


# You can also convert numpy.ndarray to connx.Tensor
connx.Tensor.from_nparray(ndarray)
```

# ONNX compatibility test
ONNX compatibility test is moved to onnx-connx project.

# Performance profile report
If you want to profile performance, you need to compile CONNX in debugging mode first.

~~~sh
connx/build$ cmake ../ports/linux -G Ninja -DCMAKE_BUILD_TYPE=Debug  # Generate build files
connx/build$ ninja                                                   # Compile
connx/build$ ninja mnist                                             # Run an any example
connx/build$ ninja prof                                              # Print performance profile report
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
