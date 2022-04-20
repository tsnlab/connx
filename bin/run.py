import struct
import sys
import time

import numpy as np

from connx import load_data, load_model


WARN = '\033[93m'
END = '\033[0m'


def get_nptype(onnx_dtype):
    if onnx_dtype == 1:
        return np.float32
    elif onnx_dtype == 2:
        return np.uint8
    elif onnx_dtype == 3:
        return np.int8
    elif onnx_dtype == 4:
        return np.uint16
    elif onnx_dtype == 5:
        return np.int16
    elif onnx_dtype == 6:
        return np.int32
    elif onnx_dtype == 7:
        return np.int64
    elif onnx_dtype == 8:
        return str
    elif onnx_dtype == 9:
        return bool
    elif onnx_dtype == 10:
        return np.float16
    elif onnx_dtype == 11:
        return np.float64
    elif onnx_dtype == 12:
        return np.uint32
    elif onnx_dtype == 13:
        return np.uint64
    elif onnx_dtype == 14:
        return np.csingle
    elif onnx_dtype == 15:
        return np.cdouble
    else:
        raise Exception('Not supported dtype: {}'.format(onnx_dtype))


def get_dtype(numpy_type):
    if numpy_type == np.float32:
        return 1
    elif numpy_type == np.uint8:
        return 2
    elif numpy_type == np.int8:
        return 3
    elif numpy_type == np.uint16:
        return 4
    elif numpy_type == np.int16:
        return 5
    elif numpy_type == np.int32:
        return 6
    elif numpy_type == np.int64:
        return 7
    elif numpy_type == str:
        return 8
    elif numpy_type == bool:
        return 9
    elif numpy_type == np.float16:
        return 10
    elif numpy_type == np.float64:
        return 11
    elif numpy_type == np.uint32:
        return 12
    elif numpy_type == np.uint64:
        return 13
    elif numpy_type == np.csingle:
        return 14
    elif numpy_type == np.cdouble:
        return 15
    else:
        raise Exception('Not supported type: {}'.format(numpy_type))


def product(shape):
    p = 1
    for dim in shape:
        p *= dim

    return p


def run(model_path, input_paths):
    model = load_model(model_path)
    input_data = [load_data(p) for p in input_paths]
    results = model.run(input_data)

    return [data.to_nparray() for data in results]


def read_tensor(io):
    # Parse data type
    dtype, ndim = struct.unpack('=II', io.read(8))

    shape = []
    for i in range(ndim):
        shape.append(struct.unpack('=I', io.read(4))[0])

    # Parse data
    dtype = get_nptype(dtype)
    itemsize = np.dtype(dtype).itemsize
    total = product(shape)
    return np.frombuffer(io.read(itemsize * total), dtype=dtype, count=product(shape)).reshape(shape)


if __name__ == '__main__':
    start_timestamp = time.time()
    outputs = run(sys.argv[1], sys.argv[2:])
    end_timestamp = time.time()

    if outputs is not None:
        for i in range(len(outputs)):
            print('# output[{}]'.format(i))
            print(outputs[i])
            print(f'# output[i].shape = {outputs[i].shape}')

    # Print elapsed time
    print(f'Elapsed time: {end_timestamp - start_timestamp} sec')
