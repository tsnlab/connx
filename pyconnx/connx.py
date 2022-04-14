import ctypes
import math
import struct

from typing import List

import numpy

from . import bindings
from . import types

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

    def __init__(self):
        super().__init__()

    def to_nparray(self) -> numpy.ndarray:
        return numpy.frombuffer(
            ctypes.string_at(self.buffer, self.size),
            dtype=types.ConnxType(self.dtype).to_numpy()
        ).reshape(self.shape)

    @staticmethod
    def from_nparray(nparray: numpy.ndarray) -> 'Tensor':
        tensor = Tensor()
        tensor._wrapped_object = bindings.alloc_tensor(
            types.ConnxType.from_numpy(nparray.dtype),
            nparray.shape
        )
        data = nparray.tostring()
        assert(len(data) == tensor.size)

        ctypes.memmove(tensor.buffer, data, len(data))

        return tensor

    @staticmethod
    def from_bytes(data: bytes) -> 'Tensor':
        tensor = Tensor()
        format_ = '=II'
        size = struct.calcsize(format_)
        dtype, ndim = struct.unpack(format_, data[:size])
        data = data[size:]
        shape = []
        for _ in range(ndim):
            format_ = '=I'
            size = struct.calcsize(format_)
            shape.append(struct.unpack(format_, data[:size])[0])
            data = data[size:]

        tensor._wrapped_object = bindings.alloc_tensor(dtype, shape)
        assert(len(data) == tensor.size)
        ctypes.memmove(tensor._wrapped_object.buffer, data, len(data))

        return tensor

    def __len__(self):
        return math.prod(self.shape)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def data(self):
        return ctypes.cast(
            self._wrapped_object.buffer,
            ctypes.POINTER(types.ConnxType(self.dtype).to_ctypes())
        )

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
    with open(path_, 'rb') as f:
        data = f.read()
    return Tensor.from_bytes(data)
