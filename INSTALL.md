# Connx Installation

## Prerequisites
 * python3 >= 3.8  # To build templates and for python bindings
 * [poetry][]      # To setup python develop environment
 * cmake >= 3.1
 * ninja-build

[poetry]: https://pypi.org/project/poetry/

If your python version is below 3.8, refer to [Installing Python3 on Linux](https://docs.python-guide.org/starting/install3/linux/)
```sh
$ sudo apt -y install cmake ninja-build
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3
# Add export PATH="$HOME/.poetry/bin:$PATH" in .bashrc
```
## Download the source code
```
$ git clone https://github.com/tsnlab/connx.git
$ cd connx
```  

## Build
### Debug mode
```
connx$ poetry install  
```
### Release mode
~~~sh
connx$ poetry install                                                   # To install python dependencies
connx$ mkdir build; cd build                                            # Make build directory
connx/build$ cmake ../ports/linux -G Ninja -D CMAKE_BUILD_TYPE=Release  # Generate build files with "Release" mode
connx/build$ ninja                                                      # Compile
~~~

> You can find **connx** executable and **libconnx.so** library in the **connx/build** directory.

### Build process overview
Optional information for your understanding
![Build Prcess](/assets/images/onnx_build.drawio.png)
