import ctypes
import os

from typing import List

import numpy

from . import bindings

libconnx = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'libconnx.so'))


class Tensor(object):

    def __init__(self, dtype, shape, data=None):
        self._struct = bindings.Tensor()

        self.dtype = dtype
        self.shape = shape
        self.data = data  # TODO: some convert

    def to_nparray(self) -> numpy.ndarray:
        return numpy.fromstring(self.data, dtype=self.type_).reshape(self.shape)

    @staticmethod
    def from_nparray(nparray: numpy.ndarray) -> 'Tensor':
        return Tensor(nparray.dtype, nparray.shape)

    @property
    def shape(self) -> List[int]:
        return self._struct.shape[:self._struct.ndim]

    @shape.setter
    def shape(self, value):
        self._struct.shape = (ctypes.c_int * len(value))(*value)
        self._struct.ndim = len(value)

    def __getattr__(self, attr):
        if attr == 'shape':
            return self._struct.shape[:self._struct.ndim]
        else:
            return getattr(self._struct, attr)

    def __setattr__(self, attr, value):
        if attr == '_struct':
            return super().__setattr__(attr, value)

        if hasattr(self._struct, attr) and attr not in ('shape'):
            setattr(self._struct, attr, value)
        else:
            super().__setattr__(attr, value)


class ConnxModel():

    def __init__(self, model_path: str):
        # TODO: load model
        pass

    def run(self, input_data: List[Tensor]) -> List[Tensor]:
        # TODO: inference and return output
        pass


def load_model(model_path: str) -> ConnxModel:
    return ConnxModel(model_path)


def load_data(path_: str) -> Tensor:
    # TODO: load data from file
    pass
