# Install
Python3 and Python package manager, pip, must be installed beofre.

## Install virtual environment
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate

## Install libraries
pip3 install tensorflow==1.14
pip3 install torch
pip3 install onnx
pip3 install onnx-tf

## Reference
1. https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
