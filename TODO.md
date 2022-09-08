# Impelemetation of operators for famous models
Here's the list to implement that we extracted the operators from real onnx model file.  

You can find the models in official onnx-model repository.  
https://github.com/onnx/models

# Operators list to be implemented
## MNIST
- [X] Add
- [X] Conv
- [X] MatMul
- [X] MaxPool
- [X] Relu
- [X] Reshape
- [X] Sign
- [X] Tan
- [X] Sin
- [X] Sinh
- [X] Cos
- [X] Cosh

## Mobilenet
- [X] Add
- [X] BatchNormalization
- [X] Conv
- [X] GlobalAveragePool
- [X] Relu
- [X] Reshape
  
##  YOLO v4
- [X] Add
- [X] Cast (without string)
- [X] Concat
- [X] Conv
- [X] Exp
- [X] Gather
- [X] GlobalMaxPool
- [X] GreaterOrEqual
- [X] LeakyRelu
- [X] MaxPool
- [X] Mul
- [X] NonZero
- [X] Reshape
- [X] Resize
- [X] Shape
- [X] Sigmoid
- [X] Slice
- [X] Softplus
- [X] Split
- [X] Squeeze
- [X] Sub
- [X] Tanh
- [X] Tile
- [X] Transpose

### ResNet v2
- [X] Add
- [X] BatchNormalization
- [X] Conv
- [X] Gemm
- [X] GlobalAveragePool
- [X] MaxPool
- [X] Relu
- [X] Reshape