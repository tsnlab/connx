import os
import sys
import numpy as np
import onnx
from onnx import numpy_helper
from onnx import helper
import onnxruntime

import common

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

def main():
    data = common.parse_args()
    if(data == None):
        return 0

    if(data['outputs'] != None):
        model = data['model']
        for output in data['outputs']:
            intermediate_layer_value_info = helper.ValueInfoProto()
            intermediate_layer_value_info.name = output
            model.graph.output.append(intermediate_layer_value_info)

        onnx.save(model, 'tmp.onnx')
        data['onnx'] = 'tmp.onnx'

    session = onnxruntime.InferenceSession(data['onnx'])

    time_start = common.get_us()
    for i in range(0, data['loop']):
        outputs = session.run(None, data['inputs'])
    time_end = common.get_us()
    print("Time:", "{:,}".format(time_end - time_start), "us")

    if(data['outputs'] != None):
        base = len(outputs) - len(data['outputs'])
        for i in range(base, len(outputs)):
            print('*', data['outputs'][i - base], outputs[i].shape)
            print(outputs[i].shape)
            print(outputs[i])

        os.remove('tmp.onnx')
    else:
        targets = data['targets']
        if(len(targets) > 0):
            if(len(targets) != len(outputs)):
                print("output count not matching: ", len(outputs), ', expected: ', len(targets))
                return 0

            i = 0
            for key in targets.keys():
                if(not np.allclose(outputs[i], targets[key], atol=data['tolerance'], rtol=0)):
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
