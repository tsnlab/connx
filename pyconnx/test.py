import numpy
from connx import load_data, load_model

model = load_model('examples/mnist')
input_data = load_data('examples/mnist/test_data_set_0/input_0.data')
reference_data = load_data('examples/mnist/test_data_set_0/output_0.data')

[inference_data] = model.run([input_data])

reference_nparray = reference_data.to_nparray()
inrefence_nparray = inference_data.to_nparray()

assert(reference_nparray.shape == inrefence_nparray.shape)
assert(numpy.allclose(reference_nparray, inrefence_nparray))
