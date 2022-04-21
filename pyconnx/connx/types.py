import ctypes
import enum

try:
    import numpy
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class ConnxType(enum.IntEnum):
    UNDEFINED = 0,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    UINT32 = 12,
    INT32 = 6,
    UINT64 = 13,
    INT64 = 7,
    FLOAT16 = 10,
    FLOAT32 = 1,
    FLOAT64 = 11,
    STRING = 8,
    BOOL = 9,
    COMPLEX64 = 14,
    COMPLEX128 = 15,

    def to_ctypes(self):
        return self._TO_CTYPES.get(self)

    def to_numpy(self):
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is not available")
        return self._TO_NUMPY.get(self)

    @classmethod
    def from_ctypes(cls, value):
        return cls._FROM_CTYPES.get(value)

    @classmethod
    def from_numpy(cls, value):
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is not available")
        return cls._FROM_NUMPY.get(numpy.dtype(value))


ConnxType._TO_CTYPES = {
    ConnxType.UNDEFINED: None,
    ConnxType.UINT8: ctypes.c_uint8,
    ConnxType.INT8: ctypes.c_int8,
    ConnxType.UINT16: ctypes.c_uint16,
    ConnxType.INT16: ctypes.c_int16,
    ConnxType.UINT32: ctypes.c_uint32,
    ConnxType.INT32: ctypes.c_int32,
    ConnxType.UINT64: ctypes.c_uint64,
    ConnxType.INT64: ctypes.c_int64,
    ConnxType.FLOAT16: None,
    ConnxType.FLOAT32: ctypes.c_float,
    ConnxType.FLOAT64: ctypes.c_double,
    ConnxType.STRING: ctypes.c_char_p,
    ConnxType.BOOL: ctypes.c_bool,
    ConnxType.COMPLEX64: None,
    ConnxType.COMPLEX128: None,
}

ConnxType._FROM_CTYPES = {
    v: k
    for k, v in ConnxType._TO_CTYPES.items()
}

if NUMPY_AVAILABLE:

    ConnxType._TO_NUMPY = {
        ConnxType.UNDEFINED: None,
        ConnxType.UINT8: numpy.uint8,
        ConnxType.INT8: numpy.int8,
        ConnxType.UINT16: numpy.uint16,
        ConnxType.INT16: numpy.int16,
        ConnxType.UINT32: numpy.uint32,
        ConnxType.INT32: numpy.int32,
        ConnxType.UINT64: numpy.uint64,
        ConnxType.INT64: numpy.int64,
        ConnxType.FLOAT16: numpy.float16,
        ConnxType.FLOAT32: numpy.float32,
        ConnxType.FLOAT64: numpy.float64,
        ConnxType.STRING: None,
        ConnxType.BOOL: bool,
        ConnxType.COMPLEX64: numpy.complex64,
        ConnxType.COMPLEX128: numpy.complex128,
    }

    ConnxType._FROM_NUMPY = {
        numpy.dtype(v) if v else None: k
        for k, v in ConnxType._TO_NUMPY.items()
    }
