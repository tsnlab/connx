idf_component_register(SRCS "main.c"
                       "../gen/tensor.c"
                       "../gen/opset.c"
                       "../gen/connx.c"
                       "../gen/accel.c"
                       "../gen/hal.c"
                       "../gen/opset/Asin.c"
                       "../gen/opset/MatMul.c"
                       "../gen/opset/Add.c"
                       "../gen/opset/Mul.c"
                       "../gen/opset/Sub.c"
                       "../gen/opset/Reshape.c"
                       "../gen/opset/Relu.c"
                       INCLUDE_DIRS "../../../include")

spiffs_create_partition_image(storage ../spiffs FLASH_IN_PROJECT)
