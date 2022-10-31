### ONNX Operators
https://github.com/onnx/onnx/blob/main/docs/Operators.md

|**Operator**               |**Since version**  |**Done**    		|
|---------------------------|-------------------|-            		|
|Abs                        |13, 6, 1           |O            		|
|Acos                       |7                  |X            		|
|Acosh                      |9                  |X            		|
|Add                        |14, 13, 7, 6, 1    |O            		|
|And                        |7, 1               |X            		|
|ArgMax                     |13, 12, 11, 1      |X            		|
|ArgMin                     |13, 12, 11, 1      |X            		|
|Asin                       |7                  |O            		|
|Asinh                      |9                  |X            		|
|Atan                       |7                  |X            		|
|Atanh                      |9                  |X            		|
|AveragePool                |11, 10, 7, 1       |X            		|
|BatchNormalization         |15, 14, 9, 7, 6, 1 |O            		|
|BitShift                   |11                 |X            		|
|Cast                       |13, 9, 6, 1        |O without stirng   |
|Ceil                       |13, 6, 1           |X            		|
|Clip                       |13, 12, 11, 6, 1   |O            		|
|Col2Im                     |18                 |X            		|
|Compress                   |11, 9              |X            		|
|Concat                     |13, 11, 4, 1       |O            		|
|ConcatFromSequence         |11                 |X            		|
|Constant                   |13, 12, 11, 9, 1   |X            		|
|ConstantOfShape            |9                  |X            		|
|Conv                       |11, 1              |O            		|
|ConvInteger                |10                 |X            		|
|ConvTranspose              |11, 1              |X            		|
|Cos                        |7                  |O            		|
|Cosh                       |9                  |O            		|
|CumSum                     |14, 11             |X            		|
|DFT                        |17                 |X            		|
|DepthToSpace               |13, 11, 1          |X            		|
|DequantizeLinear           |13, 10             |X            		|
|Det                        |11                 |X            		|
|Div                        |14, 13, 7, 6, 1    |O            		|
|Dropout                    |13, 12, 10, 7, 6, 1|X            		|
|Einsum                     |12                 |X            		|
|Elu                        |6, 1               |X            		|
|Equal                      |13, 11, 7, 1       |O            		|
|Erf                        |13, 9              |X            		|
|Exp                        |13, 6, 1           |O            		|
|Expand                     |13, 8              |X            		|
|EyeLike                    |9                  |X            		|
|Flatten                    |13, 11, 9, 1       |X            		|
|Floor                      |13, 6, 1           |X            		|
|GRU                        |14, 7, 3, 1        |X            		|
|Gather                     |13, 11, 1          |O            		|
|GatherElements             |13, 11             |X            		|
|GatherND                   |13, 12, 11         |X            		|
|Gemm                       |13, 11, 9, 7, 6, 1 |X            		|
|GlobalAveragePool          |1                  |O            		|
|GlobalLpPool               |2, 1               |X            		|
|GlobalMaxPool              |1                  |O            		|
|Greater                    |13, 9, 7, 1        |O            		|
|GridSample                 |16                 |X            		|
|HardSigmoid                |6, 1               |X            		|
|Hardmax                    |13, 11, 1          |X            		|
|Identity                   |16, 14, 13, 1      |O            		|
|If                         |16, 13, 11, 1      |X            		|
|InstanceNormalization      |6, 1               |X            		|
|IsInf                      |10                 |X            		|
|IsNaN                      |13, 9              |X            		|
|LRN                        |13, 1              |X            		|
|LSTM                       |14, 7, 1           |X            		|
|LeakyRelu                  |16, 6, 1           |O            		|
|Less                       |13, 9, 7, 1        |O            		|
|Log                        |13, 6, 1           |O            		|
|Loop                       |16, 13, 11, 1      |X            		|
|LpNormalization            |1                  |X            		|
|LpPool                     |11, 2, 1           |X            		|
|MatMul                     |13, 9, 1           |O            		|
|MatMulInteger              |10                 |X            		|
|Max                        |13, 12, 8, 6, 1    |X            		|
|MaxPool                    |12, 11, 10, 8, 1   |O            		|
|MaxRoiPool                 |1                  |X            		|
|MaxUnpool                  |11, 9              |X            		|
|Mean                       |13, 8, 6, 1        |X            		|
|MelWeightMatrix            |17                 |X            		|
|Min                        |13, 12, 8, 6, 1    |X            		|
|Mod                        |13, 10             |X            		|
|Mul                        |14, 13, 7, 6, 1    |O            		|
|Multinomial                |7                  |X            		|
|Neg                        |13, 6, 1           |X            		|
|NonMaxSuppression          |11, 10             |X            		|
|NonZero                    |13, 9              |O            		|
|Not                        |1                  |X            		|
|OneHot                     |11, 9              |X            		|
|Optional                   |15                 |X            		|
|OptionalGetElement         |18, 15             |X            		|
|OptionalHasElement         |18, 15             |X            		|
|Or                         |7, 1               |X            		|
|PRelu                      |16, 9, 7, 6, 1     |X            		|
|Pad                        |18, 13, 11, 2, 1   |X            		|
|Pow                        |15, 13, 12, 7, 1   |X            		|
|QLinearConv                |10                 |X            		|
|QLinearMatMul              |10                 |X            		|
|QuantizeLinear             |13, 10             |X            		|
|RNN                        |14, 7, 1           |X            		|
|RandomNormal               |1                  |X            		|
|RandomNormalLike           |1                  |X            		|
|RandomUniform              |1                  |X            		|
|RandomUniformLike          |1                  |X            		|
|Reciprocal                 |13, 6, 1           |X            		|
|ReduceL1                   |13, 11, 1          |X            		|
|ReduceL2                   |13, 11, 1          |X            		|
|ReduceLogSum               |13, 11, 1          |X            		|
|ReduceLogSumExp            |13, 11, 1          |X            		|
|ReduceMax                  |13, 12, 11, 1      |X            		|
|ReduceMean                 |13, 11, 1          |X            		|
|ReduceMin                  |13, 12, 11, 1      |X            		|
|ReduceProd                 |13, 11, 1          |X            		|
|ReduceSum                  |13, 11, 1          |X            		|
|ReduceSumSquare            |13, 11, 1          |X            		|
|Relu                       |14, 13, 6, 1       |O            		|
|Reshape                    |14, 13, 5, 1       |O            		|
|Resize                     |18, 13, 11, 10     |O            		|
|ReverseSequence            |10                 |X            		|
|RoiAlign                   |16, 10             |X            		|
|Round                      |11                 |X            		|
|STFT                       |17                 |X            		|
|Scan                       |16, 11, 9, 8       |X            		|
|Scatter(deprecated)        |11, 9              |X            		|
|ScatterElements            |16, 13, 11         |X            		|
|ScatterND                  |16, 13, 11         |X            		|
|Selu                       |6, 1               |X            		|
|SequenceAt                 |11                 |X            		|
|SequenceConstruct          |11                 |X            		|
|SequenceEmpty              |11                 |X            		|
|SequenceErase              |11                 |X            		|
|SequenceInsert             |11                 |X            		|
|SequenceLength             |11                 |X            		|
|Shape                      |15, 13, 1          |O            		|
|Shrink                     |9                  |X            		|
|Sigmoid                    |13, 6, 1           |O            		|
|Sign                       |13, 9              |O            		|
|Sin                        |7                  |O            		|
|Sinh                       |9                  |O            		|
|Size                       |13, 1              |X            		|
|Slice                      |13, 11, 10, 1      |O            		|
|Softplus                   |1                  |O            		|
|Softsign                   |1                  |X            		|
|SpaceToDepth               |13, 1              |X            		|
|Split                      |13, 11, 2, 1       |O            		|
|SplitToSequence            |11                 |X            		|
|Sqrt                       |13, 6, 1           |X            		|
|Squeeze                    |13, 11, 1          |O            		|
|StringNormalizer           |10                 |X            		|
|Sub                        |14, 13, 7, 6, 1    |O            		|
|Sum                        |13, 8, 6, 1        |X            		|
|Tan                        |7                  |X            		|
|Tanh                       |13, 6, 1           |O            		|
|TfIdfVectorizer            |9                  |X            		|
|ThresholdedRelu            |10                 |X            		|
|Tile                       |13, 6, 1           |O            		|
|TopK                       |11, 10, 1          |X            		|
|Transpose                  |13, 1              |O            		|
|Trilu                      |14                 |X            		|
|Unique                     |11                 |X            		|
|Unsqueeze                  |13, 11, 1          |X            		|
|Upsample (deprecated)      |10, 9, 7           |X            		|
|Where                      |16, 9              |X            		|
|Xor                        |7, 1               |O            		|
|**Function**               |**Since version**  |X            		|
|Bernoulli                  |15                 |X            		|
|BlackmanWindow             |17                 |X            		|
|CastLike                   |15                 |X            		|
|Celu                       |12                 |X            		|
|CenterCropPad              |18                 |X            		|
|DynamicQuantizeLinear      |11                 |X            		|
|GreaterOrEqual             |16, 12             |O            		|
|HammingWindow              |17                 |X            		|
|HannWindow                 |17                 |X            		|
|HardSwish                  |14                 |X            		|
|LayerNormalization         |17                 |X            		|
|LessOrEqual                |16, 12             |O            		|
|LogSoftmax                 |13, 11, 1          |X            		|
|MeanVarianceNormalization  |13, 9              |X            		|
|Mish                       |18                 |X            		|
|NegativeLogLikelihoodLoss  |13, 12             |X            		|
|Range                      |11                 |X            		|
|SequenceMap                |17                 |X            		|
|Softmax                    |13, 11, 1          |X            		|
|SoftmaxCrossEntropyLoss    |13, 12             |X            		|
