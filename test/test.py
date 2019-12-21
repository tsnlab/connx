#Ref: https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowImport.ipynb
#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()
import sys
import time
import numpy
import tensorflow as tf
import onnx
from onnx import numpy_helper
from onnx_tf.backend import prepare
numpy.set_printoptions(threshold=sys.maxsize, suppress=True)

#import matplotlib.pyplot as plt
#tf.enable_eager_execution()

print("* input");
input = onnx.load_tensor('../examples/mnist/test_data_set_0/input_0.pb')
input = numpy_helper.to_array(input)
print(input);
#plt.imshow(input.reshape(28, 28))
#plt.show()

print("* output")
output = onnx.load_tensor('../examples/mnist/test_data_set_0/output_0.pb')
output = numpy_helper.to_array(output)
print(output)

model_onnx = onnx.load('../examples/mnist/model.onnx')
model = prepare(model_onnx)

print("* model.inputs")
print(model.inputs) # Input nodes to the model
print("* model.outputs")
#model.outputs = ["Convolution110_Output_0"]
print(model.outputs) # Output nodes from the model
print("* model.tensor_dict")
print(model.tensor_dict) # All nodes in the model

#print(model.tensor_dict)
print("* model")
print(model)

print("* run")
time_start = int(round(time.time() * 1000))
for i in range(0, 1000):
    output = model.run(input)
time_end = int(round(time.time() * 1000))
print(output)
print(time_end - time_start, " ms")
#print(output.Convolution28_Output_0.shape)
#print(output.Convolution28_Output_0)
#print(output.Plus214_Output_0)
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print("#####")
#    print(sess.run(model.tensor_dict["Parameter194"]))
#    print("#####")
