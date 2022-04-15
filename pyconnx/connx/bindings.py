import os
from ctypes import (
    addressof,
    c_char_p,
    c_int32,
    c_long,
    c_uint32,
    c_void_p,
    CDLL, POINTER, Structure)


libconnx = CDLL(os.path.join(os.path.dirname(__file__), 'libconnx.so'))


class ConnxTensor(Structure):
    """
    typedef struct _connx_Tensor {
    uint32_t dtype;               // data type (enum connx_DataType)
    int32_t ndim;                 // Number of dimensions
    int32_t* shape;               // Shape array
    void* buffer;                 // Data buffer
    uint32_t size;                // size of buffer
    struct _connx_Tensor* parent; // Parent tensor that share the buffer
    int32_t ref_count;            // Reference count
    int32_t child_count;          // Child count
    connx_Lock lock;              // Reference and child count lock
    } connx_Tensor;
    """
    # Because it contains self pointer, pass here and monkey patch below
    pass


ConnxTensor._fields_ = [
    ('dtype', c_uint32),  # data type
    ('ndim', c_int32),  # Number of dimensions
    ('shape', POINTER(c_int32)),  # Shape array
    ('buffer', c_void_p),  # Data buffer
    ('size', c_uint32),  # size of buffer
    ('parent', POINTER(ConnxTensor)),  # Parent tensor that share the buffer
    ('ref_count', c_int32),  # Reference count
    ('child_count', c_int32),  # Child count
    ('lock', c_long),  # Reference and child count lock
]


class ConnxNode(Structure):
    """
    typedef struct _connx_Node {
        uint32_t output_count;
        uint32_t* outputs;

        uint32_t input_count;
        uint32_t* inputs;

        uint32_t attribute_count;
        void** attributes;

        char* op_type;
        CONNX_OPERATOR op;
    } connx_Node;
    """
    _fields_ = [
        ('output_count', c_uint32),
        ('outputs', POINTER(c_uint32)),
        ('input_count', c_uint32),
        ('inputs', POINTER(c_uint32)),
        ('attribute_count', c_uint32),
        ('attributes', POINTER(c_void_p)),
        ('op_type', c_char_p),
        ('op', c_void_p),
    ]


class ConnxGraph(Structure):
    """
    struct _connx_Graph {
    connx_Model* model;

    uint32_t id;

    uint32_t initializer_count;
    connx_Tensor** initializers;
    uint32_t input_count;
    uint32_t* inputs;

    uint32_t output_count;
    uint32_t* outputs;

    uint32_t value_info_count;
    connx_Tensor** value_infos;

    uint32_t node_count;
    connx_Node** nodes;
};
    """
    _fields_ = [
        ('model', c_void_p),
        ('id', c_uint32),
        ('initializer_count', c_uint32),
        ('initializers', POINTER(POINTER(ConnxTensor))),
        ('input_count', c_uint32),
        ('inputs', POINTER(c_uint32)),
        ('output_count', c_uint32),
        ('outputs', POINTER(c_uint32)),
        ('value_info_count', c_uint32),
        ('value_infos', POINTER(POINTER(ConnxTensor))),
        ('node_count', c_uint32),
        ('nodes', POINTER(POINTER(ConnxNode))),
    ]


class ConnxModel(Structure):
    """
    typedef struct _connx_Model {
    int32_t version;

    uint32_t opset_count;
    char** opset_names;
    uint32_t* opset_versions;

    uint32_t graph_count;
    connx_Graph** graphs;
    } connx_Model;
    """
    _fields_ = [
        ('version', c_int32),
        ('opset_count', c_uint32),
        ('opset_names', POINTER(c_char_p)),
        ('opset_versions', POINTER(c_uint32)),
        ('graph_count', c_uint32),
        ('graphs', POINTER(POINTER(ConnxGraph)))
    ]

    def __del__(self):
        model_destroy(self)


model_init = libconnx.connx_Model_init
model_init.argtypes = [POINTER(ConnxModel)]

model_destroy = libconnx.connx_Model_destroy
model_destroy.argtypes = [POINTER(ConnxModel)]
model_destroy.restype = c_int32

hal_set_model = libconnx.hal_set_model
hal_set_model.argtypes = [c_char_p]

tensor_unref = libconnx.connx_Tensor_unref
tensor_unref.argtypes = [POINTER(ConnxTensor)]
tensor_unref.restype = c_int32

libconnx.connx_Tensor_alloc.argtypes = [c_uint32, c_int32, POINTER(c_int32)]
libconnx.connx_Tensor_alloc.restype = POINTER(ConnxTensor)


def alloc_tensor(dtype, shape):
    """
    Allocate a tensor.
    """
    ndim = len(shape)
    shape_array = (c_int32 * ndim)(*shape)
    tensor_pointer = libconnx.connx_Tensor_alloc(dtype, ndim, shape_array)
    return ConnxTensor.from_address(addressof(tensor_pointer.contents))


model_run = libconnx.connx_Model_run
model_run.argtypes = [
    POINTER(ConnxModel),  # model
    c_uint32,  # input_count
    POINTER(POINTER(ConnxTensor)),  # inputs
    POINTER(c_uint32),  # output_count
    POINTER(POINTER(ConnxTensor)),  # outputs
]
model_run.restype = c_int32


libconnx.connx_init()
