idf_component_register(SRCS "main.c" "hal.c"
                    "src_gen/tensor.c" "src_gen/tensor.c" "src_gen/opset.c" "src_gen/connx.c" "src_gen/accel.c"
                    "src_gen/opset/Asin.c" "src_gen/opset/MatMul.c" "src_gen/opset/Add.c" "src_gen/opset/Mul.c" "src_gen/opset/Sub.c"
                    "src_gen/opset/Reshape.c" "src_gen/opset/Relu.c"
                    INCLUDE_DIRS "include")
