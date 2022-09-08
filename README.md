CONNX is ONNX(Open Neural Network Exchange) Runtime.  

Sections:
- [Notice](#notice)
- [CONNX](#connx)
- [Installation instructions](#installation-instructions)
- [Features](#features)
- [Usage](#usage)
- [How to use](#how-to-use)
  - [CONNX running process overview](#connx-running-process-overview)
  - [Run examples](#run-examples)
- [Run MNIST example.](#run-mnist-example)
- [Run MOBILENET example.](#run-mobilenet-example)
- [Run YOLOV4 example.](#run-yolov4-example)
    - [Using python bindings](#using-python-bindings)
- [ONNX compatibility test](#onnx-compatibility-test)
- [Ports](#ports)
- [Contribution](#contribution)
- [Supported platforms](#supported-platforms)
- [License](#license)

# Notice
We are hiring paid employee. Please contact us: contact at tsnlab dot com

# CONNX
CONNX(**'C'** stands for C language) is a technology that implemented ONNX Runtime using C.   
It aims for a technology that can be easily ported to various embedded devices such as ESP32, Rpi3/4 and etc.
* Overall Architecture
  ![](/assets/images/CONNX_architecture.png)
  

# Installation instructions
* See [Requirements & Installation](INSTALL.md)

# Features
* Portability
  * It can be ported any platform, because it's written in C.
  * No Dependency : It runs standalone without any libs(BLAS, eigen...)
  * HAL(Hardware Abstraction Layer) : It seperates logic and platform-dependent code.
* Small Footprint
  * Operator plugin : It reduce the size of the footprint by selecting operator at the time of compile.
  * Preprocessor : **onnx-connx project** pre-processes and Strip ONNX to reduce footprint size by 0.5 to 5%.
  * CONNX format : Conversion to connx format. It's functionally identical to ONNX but simple to parse
* High Performance
  * It isolate the Tensor operation from the
* Open source 
  * See [License](#License) 
  * See [CONTRIBUTING.md](CONTRIBUTING.md)

# Usage
* Infrencing : better performance for a wide variety of ML models
* Edge ML : can be used any tiny devices

# How to use
## CONNX running process overview
1. Load the ONNX model.
2. Create the runtime to run ONNX model.
3. After feeding C-ONNX Tensor input into the runtime, the output is in C-ONNX Tensor format.

## Run examples
Run MNIST example.
 
~~~sh
# Run MNIST example.
connx/build$ poetry run ninja mnist
# Run MOBILENET example.
connx/build$ poetry run ninja mobilenet
# Run YOLOV4 example.
connx/build$ poetry run ninja yolov4
~~~
or use connx excutable
~~~sh
connx/build$ ./connx ../examples/mnist/ ../examples/mnist/test_data_set_1/input_0.data
~~~
or use python script
~~~
connx$ python3 bin/run.py examples/mnist/ examples/mnist/test_data_set_1/input_0.data
~~~

> Notice: If you want to run on Raspberry Pi 3, please compile with Release mode(CMAKE\_BUILD\_TYPE=Release) for sanitizer makes some problem.

### Using python bindings

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

Please refer bin/run.py for more information

# ONNX compatibility test
ONNX compatibility test is moved to onnx-connx project.

# Ports
 * See [Linux](ports/linux/README.md)
 * See [ESP32](ports/esp32/README.md)
![](/assets/images/esp32_test.png)  
 * See [ZYNQ](ports/zynq/README.md)

# Contribution
See [CONTRIBUTING.md](CONTRIBUTING.md)

# Supported platforms
 * x86\_64
 * x86 - with CLFAGS=-m32
 * Raspberry pi 4 (w/ 64-bit O/S)
 * Raspberry pi 3 (32-bit O/S)
 * ESP32 (No O/S, firmware)
 * ZYNQ (No O/S, firmware)

# License
CONNX is licensed under GPLv3. See [LICENSE](LICENSE)
If you need other license than GPLv3 for proprietary use or professional support, please mail us to contact at tsnlab dot com.
