import numpy
import onnx
from onnx import numpy_helper
import onnxruntime

import common

def main():
    data = common.parse_args()
    if(data == None):
        return 0

    session = onnxruntime.InferenceSession(data['onnx'])

    time_start = common.get_us()
    for i in range(0, data['loop']):
        outputs = session.run(None, data['inputs'])
    time_end = common.get_us()
    print("Time:", "{:,}".format(time_end - time_start), "us")

    targets = data['targets']
    if(len(targets) > 0):
        if(len(targets) != len(outputs)):
            print("output count not matching: ", len(outputs), ', expected: ', len(targets))
            return 0

        i = 0
        for key in targets.keys():
            if(not numpy.allclose(outputs[i], targets[key], atol=data['tolerance'], rtol=0)):
                print('targets[', key, '] is not matched')
                print('* outputs[', i, ']')
                print(outputs[i])
                print('* targets[', key, ']')
                print(targets[key])
                return 0

            i = i + 1

        print('All target matched')

if __name__ == '__main__':
    main()
