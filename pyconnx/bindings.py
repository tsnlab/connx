from ctypes import c_int32, c_long, c_uint32, c_void_p, POINTER, Structure


class Tensor(Structure):
    # Because it contains self pointer, pass here and monkey patch below
    pass


Tensor._fields_ = [
    ('dtype', c_uint32),           # data type
    ('ndim', c_int32),             # Number of dimensions
    ('shape', POINTER(c_int32)),   # Shape array
    ('buffer', c_void_p),          # Data buffer
    ('size', c_uint32),            # size of buffer
    ('parent', POINTER(Tensor)),   # Parent tensor that share the buffer
    ('ref_count', c_int32),        # Reference count
    ('child_count', c_int32),      # Child count
    ('lock', c_long),              # Reference and child count lock
]
