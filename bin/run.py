import os
import sys
import subprocess
import struct
import numpy as np

def get_datatype_size(dtype):
    if dtype in [2, 3, 9]: # UINT8, INT8, BOOL
        return 1
    elif dtype in [4, 5, 10]: # UINT16, INT16, FLOAT16
        return 2
    elif dtype in [12, 6, 1]: # UINT32, INT32, FLOAT32
        return 4
    elif dtype in [13, 7, 11, 14]: # UINT64, INT64, FLOAT64, COMPLEX64
        return 8
    elif dtype == 15: # COMPLEX128
        return 16
    elif dtype == 8: # STRING (header length)
        return 4
    else:
        return 0

def product(shape):
    p = 1
    for dim in shape:
        p *= dim

    return p

def main(argv):
    connx_path = argv[1]
    model_path = argv[2]
    inputs = argv[3:]

    with subprocess.Popen([connx_path, model_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        # Write number of inputs
        proc.stdin.write(struct.pack('=I', len(inputs)))

        for input in inputs:
            tokens = os.path.basename(input).strip('.data').split('_')
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
            with open(input, 'rb') as file:
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
            if dtype == 2:
                dtype = np.uint8
            elif dtype == 3:
                dtype = np.int8
            elif dtype == 9:
                dtype = bool
            elif dtype == 4:
                dtype = np.uint16
            elif dtype == 5:
                dtype = np.int16
            elif dtype == 10:
                dtype = np.float16
            elif dtype == 12:
                dtype = np.uint32
            elif dtype == 6:
                dtype = np.int32
            elif dtype == 1:
                dtype = np.float32
            elif dtype == 13:
                dtype = np.uint64
            elif dtype == 7:
                dtype = np.int64
            elif dtype == 11:
                dtype = np.float64
            elif dtype == 14:
                dtype = np.csingle
            elif dtype == 15:
                dtype = np.cdouble
            elif dtype == 8:
                dtype = str
            else:
                raise Exception('Not supported dtype: {}'.format(dtype))

            itemsize = np.dtype(dtype).itemsize
            total = product(shape)
            output = np.frombuffer(proc.stdout.read(itemsize * total), dtype=dtype, count=product(shape)).reshape(shape)
            outputs.append(output)

        proc.stdout.close()

        return outputs

if __name__ == '__main__':
    outputs = main(sys.argv)
    for i in range(len(outputs)):
        print('# output[{}]'.format(i))
        print(outputs[i])
