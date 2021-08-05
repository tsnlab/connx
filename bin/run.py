import sys
import subprocess
import struct
import numpy as np


def get_numpy_dtype(onnx_dtype):
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


def product(shape):
    p = 1
    for dim in shape:
        p *= dim

    return p


def run(connx_path, model_path, input_paths):
    with subprocess.Popen([connx_path, model_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        # Write number of inputs
        proc.stdin.write(struct.pack('=I', len(input_paths)))

        for input_path in input_paths:
            # Write data
            with open(input_path, 'rb') as file:
                data = file.read()
                proc.stdin.write(data)

        # Terminate the connx at next loop
        proc.stdin.write(struct.pack('=i', -1))
        proc.stdin.close()

        # Parse number of outputs
        count = struct.unpack('=i', proc.stdout.read(4))[0]
        if count < 0:
            print('Error code:', count)
            return count

        outputs = []

        for i in range(count):
            # Parse data type
            dtype, ndim = struct.unpack('=II', proc.stdout.read(8))

            shape = []
            for i in range(ndim):
                shape.append(struct.unpack('=I', proc.stdout.read(4))[0])

            # Parse data
            dtype = get_numpy_dtype(dtype)
            itemsize = np.dtype(dtype).itemsize
            total = product(shape)
            output = np.frombuffer(proc.stdout.read(itemsize * total), dtype=dtype, count=product(shape)).reshape(shape)
            outputs.append(output)

        proc.stdout.close()

        return outputs


def read_tensor(io):
    # Parse data type
    dtype, ndim = struct.unpack('=II', io.read(8))

    shape = []
    for i in range(ndim):
        shape.append(struct.unpack('=I', io.read(4))[0])

    # Parse data
    dtype = get_numpy_dtype(dtype)
    itemsize = np.dtype(dtype).itemsize
    total = product(shape)
    return np.frombuffer(io.read(itemsize * total), dtype=dtype, count=product(shape)).reshape(shape)


if __name__ == '__main__':
    outputs = run(sys.argv[1], sys.argv[2], sys.argv[3:])

    if outputs is not None:
        for i in range(len(outputs)):
            print('# output[{}]'.format(i))
            print(outputs[i])
