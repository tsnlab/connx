import sys
import subprocess
import struct
import numpy as np
import threading
import traceback
import time


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


class Daemon(threading.Thread):
    def __init__(self, connx_path, model_path):
        threading.Thread.__init__(self)
        self.connx_path = connx_path
        self.model_path = model_path
        self.stderr = None
        self.stdin = None
        self.stdout = None

    def run(self):
        with subprocess.Popen([self.connx_path, self.model_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as proc:

            self.stdin = proc.stdin
            self.stdout = proc.stdout
            self.stderr = proc.stderr

            for line in iter(proc.stderr.readline, b''):
                print(f'{WARN}stderr> ', line, END, flush=True)

    def inference(self, inputs):
        while self.stderr is None:
            time.sleep(0)

        if inputs is None:
            # Terminate the connx at next loop
            self.stdin.write(struct.pack('=i', -1))
            self.stdin.flush()
            self.stdin.close()
            self.stderr.close()
            return None

        # Write number of inputs
        self.stdin.write(struct.pack('=I', len(inputs)))

        for input in inputs:
            # Write data
            if type(input) == str:
                with open(input, 'rb') as file:
                    data = file.read()
                    self.stdin.write(data)
                    self.stdin.flush()
            elif type(input) == np.ndarray:
                dtype = get_dtype(input.dtype)
                self.stdin.write(struct.pack('=I', dtype))
                self.stdin.write(struct.pack('=I', len(input.shape)))

                for dim in input.shape:
                    self.stdin.write(struct.pack('=I', dim))

                data = input.tobytes()
                self.stdin.write(data)
                self.stdin.flush()
            else:
                raise Exception(f'Unknown input type: {type(input)}')

        # save bytes for debugging purpose
        buf = bytearray()

        # Parse number of outputs
        try:
            b = self.stdout.read(4)
            buf += b
            count = struct.unpack('=i', b)[0]

            if count < 0:
                print('Error code:', count)
                return []

            outputs = []

            for i in range(count):
                # Parse data type
                b = self.stdout.read(8)
                buf += b
                dtype, ndim = struct.unpack('=II', b)

                shape = []
                for i in range(ndim):
                    b = self.stdout.read(4)
                    buf += b
                    shape.append(struct.unpack('=I', b)[0])

                # Parse data
                dtype = get_nptype(dtype)
                itemsize = np.dtype(dtype).itemsize
                total = product(shape)
                b = self.stdout.read(itemsize * total)
                buf += b
                output = np.frombuffer(b, dtype=dtype, count=product(shape)).reshape(shape)
                outputs.append(output)

            return outputs
        except Exception as e:
            print(f'Exception occurred: {e}')
            traceback.print_exc()
            print(f'Illegal input: {len(buf)} bytes')
            print(f'as string: "{buf}"')
            print(f'as hexa: "{buf.hex()}"')
            return []

    def stop(self):
        self.stdout.close()


class Stderr(threading.Thread):
    def __init__(self, stderr):
        threading.Thread.__init__(self)

        self.stderr = stderr

    def run(self):
        for line in iter(self.stderr.readline, b''):
            print(f'{WARN}stderr> ', line, END, flush=True)


def run(connx_path, model_path, inputs):
    with subprocess.Popen([connx_path, model_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as proc:

        # Write number of inputs
        proc.stdin.write(struct.pack('=I', len(inputs)))

        for input in inputs:
            # Write data
            if type(input) == str:
                with open(input, 'rb') as file:
                    data = file.read()
                    proc.stdin.write(data)
            elif type(input) == np.ndarray:
                dtype = get_dtype(input.dtype)
                proc.stdin.write(struct.pack('=I', dtype))
                proc.stdin.write(struct.pack('=I', len(input.shape)))

                for dim in input.shape:
                    proc.stdin.write(struct.pack('=I', dim))

                data = input.tobytes()
                proc.stdin.write(data)
            else:
                raise Exception(f'Unknown input type: {type(input)}')

        # Terminate the connx at next loop
        proc.stdin.write(struct.pack('=i', -1))
        proc.stdin.close()

        # Print stderr first
        stderr = Stderr(proc.stderr)
        stderr.start()

        # save bytes for debugging purpose
        buf = bytearray()

        # Parse number of outputs
        try:
            b = proc.stdout.read(4)
            buf += b
            count = struct.unpack('=i', b)[0]

            if count < 0:
                print('Error code:', count)
                stderr.join()
                return []

            outputs = []

            for i in range(count):
                # Parse data type
                b = proc.stdout.read(8)
                buf += b
                dtype, ndim = struct.unpack('=II', b)

                shape = []
                for i in range(ndim):
                    b = proc.stdout.read(4)
                    buf += b
                    shape.append(struct.unpack('=I', b)[0])

                # Parse data
                dtype = get_nptype(dtype)
                itemsize = np.dtype(dtype).itemsize
                total = product(shape)
                b = proc.stdout.read(itemsize * total)
                buf += b
                output = np.frombuffer(b, dtype=dtype, count=product(shape)).reshape(shape)
                outputs.append(output)

            proc.stdout.close()
        except Exception as e:
            print(f'Exception occurred: {e}')
            traceback.print_exc()
            print(f'Illegal input: {len(buf)} bytes')
            print(f'as string: "{buf}"')
            print(f'as hexa: "{buf.hex()}"')
            return []

        stderr.join()

        return outputs


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
    outputs = run(sys.argv[1], sys.argv[2], sys.argv[3:])

    if outputs is not None:
        for i in range(len(outputs)):
            print('# output[{}]'.format(i))
            print(outputs[i])
