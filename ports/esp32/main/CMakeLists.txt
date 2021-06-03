idf_component_register(SRCS "main.c"
                       "../gen/tensor.c"
                       "../gen/opset.c"
                       "../gen/connx.c"
                       "../gen/accel.c"
                       "../gen/hal.c"
                       INCLUDE_DIRS "../../../include")

spiffs_create_partition_image(storage ../spiffs FLASH_IN_PROJECT)
