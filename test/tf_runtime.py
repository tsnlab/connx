import sys
import getopt
import time
import numpy
import tensorflow as tf
import onnx
from onnx import numpy_helper
from onnx_tf.backend import prepare
numpy.set_printoptions(threshold=sys.maxsize, suppress=True)

TITLE = "CONNX - C implementation of Open Neural Network Exchange Runtime"
COPYRIGHT_HOLDER = "Semih Kim"

def get_us():
    return int(round(time.time() * 1000000))

def print_notice():
    print("CONNX  Copyright (C) 2019  ", COPYRIGHT_HOLDER, "\n")
    print("This program comes with ABSOLUTELY NO WARRANTY.")
    print("This is free software, and you are welcome to redistribute it")
    print("under certain conditions; use -c option for details.\n")

def print_copyright():
    print(TITLE, "")
    print("Copyright (C) 2019  ", COPYRIGHT_HOLDER, "\n")

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
    print("\tconnx [onnx file] -i [input data] [-t [target data]] [-l [loop count]] [-d]\n")
    print("Options:")
    print("\t-i    Input data file (protocol buffer format)")
    print("\t-t    Target data file (protocol buffer format)")
    print("\t-o    Output variable")
    print("\t-l    Loop count (default is 1)")
    print("\t-d    Dump variables")
    print("\t-h    Display this help message")
    print("\t-v    Display this application version")
    print("\t-c    Display copyright")

def print_version():
    print("CONNX ver 0.0.0")

def main():
    if(len(sys.argv) < 2):
        print_notice();
        print_help();
        return 1

    fileOnnx = sys.argv[1]
    fileInput = None
    fileTarget = None
    outputVariable = None
    loopCount = 1
    isDebug = False

    try:
        opts, args = getopt.getopt(sys.argv[2:], "i:t:l:dhvc")
    except getopt.GetoptError as err:
        print('error', err)
        print_help()
        return 1

    for opt, arg in opts:
        if(opt == "-i"):
            fileInput = arg
        elif(opt == "-t"):
            fileTarget = arg
        elif(opt == "-o"):
            outputVariable = arg
        elif(opt == "-l"):
            loopCount = int(arg)
        elif(opt == "-d"):
            isDebug = True
        elif(opt == "-h"):
            print_notice();
            print_help();
            return 0;
        elif(opt == "-v"):
            print_notice();
            print_version();
            return 0;
        elif(opt == "-c"):
            print_copyright();
            return 0;
        else:
            print_notice();
            print_help();
            return 1;

    print_notice()

    if(fileInput == None):
        print("You must specify input data.")
        print_help();
        return 1;

    input = onnx.load_tensor(fileInput)
    input = numpy_helper.to_array(input)
    if(isDebug):
        print("* input")
        print(input);

    model_onnx = onnx.load(fileOnnx)
    model = prepare(model_onnx)
    if(isDebug):
        print("* model")
        print(model)

    if(outputVariable != None):
        model.outputs = [ outputVariable ]

    time_start = get_us()
    for i in range(0, loopCount):
        result = model.run(input)
    time_end = get_us()
    print("Time:", "{:,}".format(time_end - time_start), "us")

    if(result != None):
        print(result)

        if(fileTarget != None):
            result2 = result[model.outputs[0]]

            target = onnx.load_tensor(fileTarget)
            target = numpy_helper.to_array(target)

            if((result2 == target).all()):
                print("Target matched")
            else:
                print("Target not matched")
                print(target);

if __name__ == '__main__':
    main()
