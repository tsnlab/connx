import ctypes
import os

from typing import List

import numpy

from . import bindings

__all__ = (
    'ConnxModel',
    'Tensor',
    'load_model',
    'load_data',
)


class Wrapper(object):

    def __init__(self):
        self._wrapped_object = self._wrapped_class_()
        self._as_parameter_ = self._wrapped_object

    def __getattr__(self, attr):
        return getattr(self._wrapped_object, attr)

    def __setattr__(self, attr, value):
        if attr == '_wrapped_object':
            super().__setattr__(attr, value)
        elif hasattr(self, attr):
            super().__setattr__(attr, value)
        else:
            setattr(self._wrapped_object, attr, value)

    def __dir__(self):
        return super().__dir__() + dir(self._wrapped_object)

    def __del__(self):
        del self._wrapped_object


class Tensor(Wrapper):

    _wrapped_class_ = bindings.ConnxTensor

    def __init__(self, dtype, shape, data=None):
        super().__init__()

        self.dtype = dtype
        self.shape = shape
        self.data = data  # TODO: some convert

    def to_nparray(self) -> numpy.ndarray:
        return numpy.fromstring(self.data, dtype=self.type_).reshape(self.shape)

    @staticmethod
    def from_nparray(nparray: numpy.ndarray) -> 'Tensor':
        return Tensor(nparray.dtype, nparray.shape)  # TODO: convert data

    @property
    def shape(self) -> List[int]:
        return self._wrapped_object.shape[:self._wrapped_object.ndim]

    @shape.setter
    def shape(self, value):
        self._wrapped_object.shape = (ctypes.c_int * len(value))(*value)
        self._wrapped_object.ndim = len(value)


class ConnxModel(Wrapper):

    _wrapped_class_ = bindings.ConnxModel

    def __init__(self, model_path: str):
        super().__init__()
        bindings.hal_set_model(model_path.encode())
        bindings.model_init(self._wrapped_object)

    def run(self, input_data: List[Tensor]) -> List[Tensor]:
        # TODO: inference and return output
        pass


def load_model(model_path: str) -> ConnxModel:
    return ConnxModel(model_path)


def load_data(path_: str) -> Tensor:
    # TODO: load data from file
    pass
