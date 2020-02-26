# Install
Python3 and Python package manager, pip, must be installed beofre.

## Install virtual environment
python3 -m pip install --user virtualenv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

## Install libraries
pip install onnx
pip install onnxruntime	# for Microsoft's ONNX Runtime
pip install torch		# for Facebook's PyTorch
pip install tensorflow	# for Google's TensorFlow

# Run
## Run ONNX Runtime
python rt_onnxruntime.py

## Run PyTorch
To be done

## Run TensorFlow
To be done

# Import onnx library to make test cases
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

sys.path.insert(1, '../onnx/onnx/backend/test/case/node')
from pool_op_common import *

# Reference
1. https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
2. https://pytorch.org/docs/stable/onnx.html
3. https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-convert-aml-deploy-tinyyolo.ipynb
4. https://microsoft.github.io/onnxruntime/python/api_summary.html
