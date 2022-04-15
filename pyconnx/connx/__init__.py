import ctypes
import math
import struct

from threading import Lock
from typing import List, Optional

try:
    import numpy
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from . import bindings
from . import types

__all__ = (
    'Model',
    'Tensor',
    'load_model',
    'load_data',
)

s_model_lock = Lock()  # To avoid conflict, Only one thread can load model at once


class Wrapper(object):

    def __init__(self):
        if '_wrapped_object' not in self.__dict__:
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
        return dir(super()) + dir(self._wrapped_object)

    def __del__(self):
        del self._wrapped_object


class Tensor(Wrapper):

    _wrapped_class_ = bindings.ConnxTensor

    def __init__(self, tensor: Optional[bindings.ConnxTensor] = None):
        if tensor:
            self._wrapped_object = tensor
        super().__init__()

    def to_nparray(self) -> 'numpy.ndarray':
        if not NUMPY_AVAILABLE:
            raise RuntimeError('numpy is not available')

        return numpy.frombuffer(
            ctypes.string_at(self.buffer, self.size),
            dtype=types.ConnxType(self.dtype).to_numpy()
        ).reshape(self.shape)

    @staticmethod
    def from_nparray(nparray: 'numpy.ndarray') -> 'Tensor':
        if not NUMPY_AVAILABLE:
            raise RuntimeError('numpy is not available')

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

    def __repr__(self):
        name = self.__class__.__name__
        dtype = types.ConnxType(self.dtype).name
        shape = self.shape
        return f'<{name} {dtype=} {shape=}>'

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


class Model(Wrapper):

    _wrapped_class_ = bindings.ConnxModel

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        with s_model_lock:
            bindings.hal_set_model(model_path.encode())
            bindings.model_init(self._wrapped_object)

    def run(self, input_data: List[Tensor]) -> List[Tensor]:
        inputs = (ctypes.POINTER(Tensor._wrapped_class_) * len(input_data))(
            *[ctypes.pointer(t._wrapped_object) for t in input_data])
        input_count = len(inputs)
        outputs = (ctypes.POINTER(Tensor._wrapped_class_) * 16)()
        output_count = ctypes.c_uint32(self.graphs[0].contents.output_count)

        bindings.model_run(
            self._wrapped_object,
            input_count,
            inputs,
            ctypes.byref(output_count),
            outputs)

        # TODO: unref?

        return [Tensor(outputs[i].contents) for i in range(output_count.value)]

    def __repr__(self):
        name = self.__class__.__name__
        version = self.version
        path = self.model_path
        return f'<{name} {version=} {path=}>'


def load_model(model_path: str) -> Model:
    return Model(model_path)


def load_data(path_: str) -> Tensor:
    with open(path_, 'rb') as f:
        data = f.read()
    return Tensor.from_bytes(data)
