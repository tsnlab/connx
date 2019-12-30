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

# Run onnx
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

sys.path.insert(1, '../onnx/onnx/backend/test/case/node')
from pool_op_common import *

## Reference
1. https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
