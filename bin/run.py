import os
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
        raise Exception('Not supported dtype: {}'.format(dtype))

def product(shape):
    p = 1
    for dim in shape:
        p *= dim

    return p

def run_direct(connx_path, model_path, input_paths):
    with subprocess.Popen([connx_path, model_path], 
                          stdin=subprocess.PIPE, 
                          stdout=subprocess.PIPE) as proc:
        # Write number of inputs
        proc.stdin.write(struct.pack('=I', len(input_paths)))

        for input_path in input_paths:
            tokens = os.path.basename(input_path).strip('.data').split('_')
            tokens.pop(0) # drop name

            # Write data type and ndim
            dtype = int(tokens.pop(0))
            ndim = int(tokens.pop(0))
            proc.stdin.write(struct.pack('=II', dtype, ndim))

            # Write shape
            shape = [ int(token) for token in tokens ]
            for dim in shape:
                proc.stdin.write(struct.pack('=I', dim))

            # Write data
            with open(input_path, 'rb') as file:
                data = file.read()
                proc.stdin.write(data)

        # Terminate the connx at next loop
        proc.stdin.write(struct.pack('=i', -1))
        proc.stdin.close()

        # Parse number of outputs
        count = struct.unpack('=i', proc.stdout.read(4))[0]
        outputs = [ ]

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
    
def main(argv):
    if len(argv) < 2:
        print('Usage: python run.py [tensor in pipe] [tensor out pipe] [input path]...]')
        return None

    tensorin_path = argv[1]
    tensorout_path = argv[2]

    if len(argv) == 3: # Stop the engine
        with open(tensorin_path, 'wb') as io:
            io.write(struct.pack('=i', -1))
            return None

    inputs = argv[3:]

    # Write input tensors
    with open(tensorin_path, 'wb') as io:
        # Write number of inputs
        io.write(struct.pack('=I', len(inputs)))

        for input in inputs:
            tokens = os.path.basename(input).strip('.data').split('_')
            tokens.pop(0) # drop name

            # Write data type and ndim
            dtype = int(tokens.pop(0))
            ndim = int(tokens.pop(0))
            io.write(struct.pack('=II', dtype, ndim))

            # Write shape
            shape = [ int(token) for token in tokens ]
            for dim in shape:
                io.write(struct.pack('=I', dim))

            # Write data
            with open(input, 'rb') as file:
                data = file.read()
                while len(data) != 0:
                    io.write(data)
                    data = file.read()

    # Read output tensors
    outputs = [ ]
    with open(tensorout_path, 'rb') as io:
        # Parse number of outputs
        count = struct.unpack('=i', io.read(4))[0]

        for i in range(count):
            # Parse data type
            dtype, ndim = struct.unpack('=II', io.read(8))

            shape = []
            for i in range(ndim):
                shape.append(struct.unpack('=I', io.read(4))[0])

            # Parse data
            dtype = get_numpy_dtype(dtype)
            itemsize = np.dtype(dtype).itemsize
            total = product(shape)
            output = np.frombuffer(io.read(itemsize * total), dtype=dtype, count=product(shape)).reshape(shape)
            outputs.append(output)

    return outputs

if __name__ == '__main__':
    outputs = main(sys.argv)

    if outputs is not None:
        for i in range(len(outputs)):
            print('# output[{}]'.format(i))
            print(outputs[i])

