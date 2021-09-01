/*
 *  CONNX, C implementation of Open Neural Network Exchange Runtime
 *  Copyright (C) 2019-2021 TSN Lab, Inc.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <stdio.h>
#include <string.h>

#include <sys/stat.h>

#include <connx/connx.h>

// HAL API from hal.c
int connx_set_model(const char* path);
int connx_set_tensorin(const char* path);
int connx_set_tensorout(const char* path);

static int read_tensor(connx_Tensor** tensor) {
    int32_t dtype;
    if (connx_read(&dtype, sizeof(int32_t)) != (int32_t)sizeof(int32_t)) {
        connx_error("Cannot read input data type from Tensor I/O module.\n");

        return CONNX_IO_ERROR;
    }

    int32_t ndim;
    if (connx_read(&ndim, sizeof(int32_t)) != (int32_t)sizeof(int32_t)) {
        connx_error("Cannot read input ndim from Tensor I/O module.\n");

        return CONNX_IO_ERROR;
    }

    int32_t shape[ndim];
    if (connx_read(&shape, sizeof(int32_t) * ndim) != (int32_t)(sizeof(int32_t) * ndim)) {
        connx_error("Cannot read input shape from Tensor I/O module.\n");

        return CONNX_IO_ERROR;
    }

    *tensor = connx_Tensor_alloc(dtype, ndim, shape);
    if (*tensor == NULL) {
        connx_error("Out of memory\n");

        return CONNX_NOT_ENOUGH_MEMORY;
    }

    int32_t len = connx_read((*tensor)->buffer, (*tensor)->size);
    if (len != (int32_t)(*tensor)->size) {
        connx_error("Cannot read input data from Tensor I/O moudle.\n");

        return CONNX_IO_ERROR;
    }

    return CONNX_OK;
}

static int write_tensor(connx_Tensor* tensor) {
    int ret;

    int32_t dtype = tensor->dtype;
    if (connx_write(&dtype, sizeof(int32_t)) != (int32_t)sizeof(int32_t)) {
        connx_error("Cannot write tensor data type to Tensor I/O module.\n");

        ret = -CONNX_IO_ERROR;
        connx_write(&ret, sizeof(int32_t));

        return CONNX_IO_ERROR;
    }

    if (connx_write(&tensor->ndim, sizeof(int32_t)) != (int32_t)sizeof(int32_t)) {
        connx_error("Cannot write tensor ndim to Tensor I/O module.\n");

        ret = -CONNX_IO_ERROR;
        connx_write(&ret, sizeof(int32_t));

        return CONNX_IO_ERROR;
    }

    if (connx_write(tensor->shape, sizeof(int32_t) * tensor->ndim) != (int32_t)(sizeof(int32_t) * tensor->ndim)) {
        connx_error("Cannot write tensor shape to Tensor I/O module.\n");

        ret = -CONNX_IO_ERROR;
        connx_write(&ret, sizeof(int32_t));

        return CONNX_IO_ERROR;
    }

    if (connx_write(tensor->buffer, tensor->size) != (int32_t)tensor->size) {
        connx_error("Cannot write tensor data to Tensor I/O module.\n");

        ret = -CONNX_IO_ERROR;
        connx_write(&ret, sizeof(int32_t));

        return CONNX_IO_ERROR;
    }

    return CONNX_OK;
}

static int run_from_pipe(connx_Model* model, char* tensorin, char* tensorout) {
    int ret;

    if (tensorin != NULL) {
        ret = connx_set_tensorin(tensorin);
        if (ret != 0) {
            return ret;
        }
    }

    if (tensorout != NULL) {
        ret = connx_set_tensorout(tensorout);
        if (ret != 0) {
            return ret;
        }
    }

    // loop: input -> inference -> output
    // If input_count is -1 then exit the loop
    while (true) {
        // Read input count from HAL
        uint32_t input_count;
        if (connx_read(&input_count, sizeof(uint32_t)) != (int32_t)sizeof(uint32_t)) {
            connx_error("Cannot read input count from Tensor I/O module.\n");

            ret = -CONNX_IO_ERROR;
            connx_write(&ret, sizeof(int32_t));

            return CONNX_IO_ERROR;
        }

        if (input_count == (uint32_t)-1) {
            break;
        }

        // Read input data from HAL
        connx_Tensor* inputs[input_count];
        for (uint32_t i = 0; i < input_count; i++) {
            connx_Tensor* tensor;
            ret = read_tensor(&tensor);

            if (ret != CONNX_OK) {
                int code = -ret;
                connx_write(&code, sizeof(int32_t));

                return ret;
            }

            inputs[i] = tensor;
        }

        // Run model
        uint32_t output_count = 16;
        connx_Tensor* outputs[output_count];

        ret = connx_Model_run(model, input_count, inputs, &output_count, outputs);
        if (ret != CONNX_OK) {
            int32_t ret2 = -ret;
            connx_write(&ret2, sizeof(int32_t));

            return ret;
        }

        // Write outputs
        // Write output count
        if (connx_write(&output_count, sizeof(uint32_t)) != (int32_t)sizeof(uint32_t)) {
            connx_error("Cannot write output data to Tensor I/O moudle.\n");

            ret = -CONNX_IO_ERROR;
            connx_write(&ret, sizeof(int32_t));

            return CONNX_IO_ERROR;
        }

        for (uint32_t i = 0; i < output_count; i++) {
            ret = write_tensor(outputs[i]);

            if (ret != CONNX_OK) {
                int code = -ret;
                connx_write(&code, sizeof(int32_t));

                return ret;
            }
        }

        /*
        for (uint32_t i = 0; i < output_count; i++) {
            printf("***** output[%u]\n", i);
            connx_Tensor_dump(outputs[i]);
        }
        */

        for (uint32_t i = 0; i < output_count; i++) {
            connx_Tensor_unref(outputs[i]);
        }
    }

    return CONNX_OK;
}

static int run_from_file(connx_Model* model, int input_count, char** input_files) {
    int ret;

    // Read input tensors
    connx_Tensor* inputs[input_count];

    for (int i = 0; i < input_count; i++) {
        // Open file
        ret = connx_set_tensorin(input_files[i]);
        if (ret != CONNX_OK) {
            return ret;
        }

        for (int i = 0; i < input_count; i++) {
            connx_Tensor* tensor;
            ret = read_tensor(&tensor);

            if (ret != CONNX_OK) {
                connx_error("Cannot read tensor: %s\n", input_files[i]);
                return ret;
            }

            inputs[i] = tensor;
        }

        // Close file
        connx_set_tensorin(NULL);
    }

    // Run model
    uint32_t output_count = 16;
    connx_Tensor* outputs[output_count];

    ret = connx_Model_run(model, input_count, inputs, &output_count, outputs);
    if (ret != CONNX_OK) {
        connx_error("Cannot run model\n");
        return ret;
    }

    for (uint32_t i = 0; i < output_count; i++) {
        connx_Tensor_dump(outputs[i]);
    }

    return CONNX_OK;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        connx_info("Usage: connx [connx model path] [[tensor in pipe] tensor out pipe] or [input tensor file]+]\n");
        return 0;
    }

    int32_t ret = CONNX_OK;

    // Load connx model
    ret = connx_set_model(argv[1]);
    if (ret != 0) {
        return ret;
    }

    int input_type = 0; // 1: input from file, 2: input from FIFO, 3: std I/O

    // Parse args
    if (argc > 2) {
        struct stat st;
        if ((ret = stat(argv[2], &st)) != 0) {
            connx_error("Cannot get file status: %s\n", argv[2]);
            return ret;
        }

        if (S_ISREG(st.st_mode)) {
            input_type = 1;
        } else if (S_ISFIFO(st.st_mode)) {
            input_type = 2;
        } else {
            connx_error("Unknown file type: %s\n", argv[2]);
            return -1;
        }
    } else {
        input_type = 3;
    }

    // Parse connx model
    connx_Model model;
    ret = connx_Model_init(&model);
    if (ret != 0) {
        return ret;
    }

    switch (input_type) {
    case 1: // input from file
        ret = run_from_file(&model, argc - 2, argv + 2);
        break;
    case 2: // input from PIPE
        ret = run_from_pipe(&model, argv[2], argv[3]);
        break;
    case 3: // input from std I/O
        ret = run_from_pipe(&model, NULL, NULL);
        break;
    default:; // Do nothing
    }

    connx_Model_destroy(&model);

    return ret;
}
