import sys
import getopt
import time
import numpy
import onnx
from onnx import numpy_helper

# numpy.set_printoptions(threshold=sys.maxsize, suppress=True)

TITLE = "CONNX - C implementation of Open Neural Network Exchange Runtime"
COPYRIGHT_HOLDER = "Semih Kim"

def get_us():
    return int(round(time.time() * 1000000))

def print_notice():
    print("CONNX  Copyright (C) 2019-2020  ", COPYRIGHT_HOLDER, "\n")
    print("This program comes with ABSOLUTELY NO WARRANTY.")
    print("This is free software, and you are welcome to redistribute it")
    print("under certain conditions; use -c option for details.\n")

def print_copyright():
    print(TITLE, "")
    print("Copyright (C) 2019-2020  ", COPYRIGHT_HOLDER, "\n")

    print("This program is free software: you can redistribute it and/or modify")
    print("it under the terms of the GNU General Public License as published by")
    print("the Free Software Foundation, either version 3 of the License, or")
    print("(at your option) any later version.\n")

    print("This program is distributed in the hope that it will be useful,")
    print("but WITHOUT ANY WARRANTY; without even the implied warranty of")
    print("MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the")
    print("GNU General Public License for more details.\n")

    print("You should have received a copy of the GNU General Public License")
    print("along with this program.  If not, see <https://www.gnu.org/licenses/>.")

def print_help():
    print("Usage:")
    print("\t", sys.argv[0], " [onnx file] -i [input data] [-t [target data]] [-l [loop count]] [-e [tolerance number]] [-d]\n")
    print("Options:")
    print("\t-i    Input data file (protocol buffer format)")
    print("\t-t    Target data file (protocol buffer format)")
    print("\t-o    Output variable")
    print("\t-l    Loop count (default is 1)")
    print("\t-e    Tolerance number (default is 0.00001)")
    print("\t-d    Dump variables")
    print("\t-h    Display this help message")
    print("\t-v    Display this application version")
    print("\t-c    Display copyright")

def print_version():
    print("CONNX ver 0.0.0")

def parse_args():
    if(len(sys.argv) < 2):
        print_notice()
        print_help()
        return None 

    data = {
        'isDebug': False,
        'onnx': sys.argv[1],
        'model': None,
        'outputs': None,
        'loop': 1,
        'tolerance': 0.00001,
        'inputs': {},
        'targets': {},
    }

    data['model'] = onnx.load(data['onnx'])

    try:
        opts, args = getopt.getopt(sys.argv[2:], "i:t:o:l:e:dhvc")
    except getopt.GetoptError as err:
        print_help()
        return None

    for opt, arg in opts:
        if(opt == "-i"):
            tensor = onnx.load_tensor(arg)
            name = tensor.name
            idx = len(data['inputs'])
            if(name == "" and len(data['model'].graph.input) > idx):
                name = data['model'].graph.input[idx].name
            data['inputs'][name] = numpy_helper.to_array(tensor)
        elif(opt == "-t"):
            tensor = onnx.load_tensor(arg)
            name = tensor.name
            idx = len(data['targets'])
            if(name == "" and len(data['model'].graph.output) > idx):
                name = data['model'].graph.output[idx].name
            data['targets'][name] = numpy_helper.to_array(tensor)
        elif(opt == "-o"):
            data['outputs'] = [ arg ]
        elif(opt == "-l"):
            data['loop'] = int(arg)
        elif(opt == "-e"):
            data['tolerance'] = float(arg)
        elif(opt == "-d"):
            data['isDebug'] = True
        elif(opt == "-h"):
            print_notice()
            print_help()
            return None
        elif(opt == "-v"):
            print_notice()
            print_version()
            return None
        elif(opt == "-c"):
            print_copyright()
            return None
        else:
            print_notice()
            print_help()
            return None

    print_notice()

    if(data['isDebug']):
        print('* inputs')
        for key in data['inputs'].keys():
            print('\t', key, ':', data['inputs'][key].shape)

        print('* targets')
        for key in data['targets'].keys():
            print('\t', key, ':', data['targets'][key].shape)

        print('* outputs')
        print('\t', data['outputs'])

        model = data['model']
        print('* producer_name :', model.producer_name)
        print('* producer_version :', model.producer_version)
        print('* graph.name :', model.graph.name)

        print('* graph.input')
        for input in model.graph.input:
            print('\t', input.name)

        print('* graph.output')
        for output in model.graph.output:
            print('\t', output.name)

    return data
