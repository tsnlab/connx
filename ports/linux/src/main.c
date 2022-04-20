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
#define _POSIX_C_SOURCE 199309L // clock_gettime
#include <inttypes.h>
#include <locale.h> // setlocale
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> // clock_gettime

#include <sys/stat.h>

#include <connx/connx.h>
#include <connx/hal.h>

static FILE* open_tensorin(const char* path) {
    return fopen(path, "r");
}

static int close_tensor(FILE* fp) {
    return fclose(fp);
}

static int32_t file_read(FILE* fp, void* buf, int32_t size) {
    void* p = buf;
    size_t remain = size;
    while (remain > 0) {
        int len = fread(p, 1, remain, fp);

        if (len < 0) {
            fprintf(stderr, "HAL ERROR: Cannot read input data");
            return -1;
        }

        p += len;
        remain -= len;
    }

    return size;
}

static int read_tensor(FILE* fp, connx_Tensor** tensor) {
    int32_t dtype;
    if (file_read(fp, &dtype, sizeof(int32_t)) != (int32_t)sizeof(int32_t)) {
        connx_error("Cannot read input data type from Tensor I/O module.\n");

        return CONNX_IO_ERROR;
    }

    int32_t ndim;
    if (file_read(fp, &ndim, sizeof(int32_t)) != (int32_t)sizeof(int32_t)) {
        connx_error("Cannot read input ndim from Tensor I/O module.\n");

        return CONNX_IO_ERROR;
    }

    int32_t shape[ndim];
    if (file_read(fp, &shape, sizeof(int32_t) * ndim) != (int32_t)(sizeof(int32_t) * ndim)) {
        connx_error("Cannot read input shape from Tensor I/O module.\n");

        return CONNX_IO_ERROR;
    }

    *tensor = connx_Tensor_alloc(dtype, ndim, shape);
    if (*tensor == NULL) {
        connx_error("Out of memory\n");

        return CONNX_NOT_ENOUGH_MEMORY;
    }

    int32_t len = file_read(fp, (*tensor)->buffer, (*tensor)->size);
    if (len != (int32_t)(*tensor)->size) {
        connx_error("Cannot read input data from Tensor I/O moudle.\n");

        return CONNX_IO_ERROR;
    }

    return CONNX_OK;
}

static int run_from_file(connx_Model* model, int input_count, char** input_files, int loop) {
    int ret;

    // Read input tensors
    connx_Tensor* inputs[input_count];

    for (int i = 0; i < input_count; i++) {
        // Open file
        FILE* tensorin = open_tensorin(input_files[i]);
        if (tensorin == NULL) {
            return CONNX_RESOURCE_NOT_FOUND;
        }

        connx_Tensor* tensor;
        ret = read_tensor(tensorin, &tensor);

        if (ret != CONNX_OK) {
            connx_error("Cannot read tensor: %s\n", input_files[i]);
            return ret;
        }

        inputs[i] = tensor;

        // Close file
        close_tensor(tensorin);
    }

    // Run model
    uint32_t output_count = 16;
    connx_Tensor* outputs[output_count];

    if (loop > 0) { // performance test
        setlocale(LC_NUMERIC, "");

        for (int i = 0; i < input_count; i++) {
            connx_Tensor_ref_child(inputs[i]);
        }

        ret = CONNX_OK;

        uint64_t total = 0;
        uint64_t minimum = UINT64_MAX;
        uint64_t maximum = 0;
        struct timespec start = {0, 0};
        struct timespec end = {0, 0};

        for (int i = 0; i < loop && ret == CONNX_OK; i++) {
            clock_gettime(CLOCK_MONOTONIC, &start);
            ret = connx_Model_run(model, input_count, inputs, &output_count, outputs);
            clock_gettime(CLOCK_MONOTONIC, &end);

            uint64_t st = start.tv_sec * 1000000000ull + start.tv_nsec;
            uint64_t et = end.tv_sec * 1000000000ull + end.tv_nsec;
            uint64_t dt = et - st;

            total += dt;

            if (dt < minimum)
                minimum = dt;

            if (dt > maximum)
                maximum = dt;

            printf("%u: %'" PRIu64 " ns\n", i, dt);

            for (uint32_t j = 0; j < output_count; j++) {
                connx_Tensor_unref(outputs[j]);
            }
        }

        printf("\n");
        printf("minimum: %'12" PRIu64 " ns\n", minimum);
        printf("mean:    %'12" PRIu64 " ns\n", (total / loop));
        printf("maximum: %'12" PRIu64 " ns\n", maximum);
        printf("total:   %'12" PRIu64 " ns\n", total);

        for (int i = 0; i < input_count; i++) {
            connx_Tensor_unref_child(inputs[i]);
        }

        printf("\n");
        connx_watch_dump();
    } else {
        ret = connx_Model_run(model, input_count, inputs, &output_count, outputs);
        if (ret != CONNX_OK) {
            connx_error("Cannot run model\n");
            return ret;
        }

        for (uint32_t j = 0; j < output_count; j++) {
            connx_Tensor_dump(outputs[j]);
            connx_Tensor_unref(outputs[j]);
        }
    }

    return CONNX_OK;
}

static int get_file_type(char* path) {
    int ret;
    struct stat st;
    if ((ret = stat(path, &st)) != 0) {
        return -2; // error
    }

    if (S_ISREG(st.st_mode)) { // regular file
        return 1;
    } else if (S_ISFIFO(st.st_mode)) { // fifo
        connx_error("Cannot read from FIFO: %s\n", path);
        return -2;
    } else { // Unknown
        return -1;
    }
}

int main(int argc, char** argv) {
    int32_t ret = CONNX_OK;

    connx_init();

    if (argc < 2) {
        connx_info("Usage: connx [connx model path] [input]* [output]? [-p number]?\n");
        connx_info("       input  - tensor file\n");
        connx_info("       -p - performance test number of times\n");
        goto done;
    }

    // Load connx model
    ret = hal_set_model(argv[1]);
    if (ret != 0) {
        goto done;
    }

    // Parse args
    char* tmp[] = {"-"};

    int loop = 0;
    int input_count = 1;
    char** input_paths = tmp;

    // 0: both omitted
    // 1: first input
    // 2: file inputs
    // 3: fifo output
    // 9: done
    int state = 0;

    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "-p", 2) == 0) {
            i++;
            loop = strtol(argv[i++], NULL, 10);
            continue;
        }

        switch (state) {
        case 0:
            state = 1;
            i--;
            break;
        case 1: // first input
            switch (get_file_type(argv[i])) {
            case -2: // I/O error
                connx_error("Cannot get file state: '%s'\n", argv[i]);
                return CONNX_IO_ERROR;
            case -1: // Unsupported file type
                connx_error("Unknown file type: '%s'\n", argv[i]);
                return CONNX_IO_ERROR;
            case 1: // regular file
                input_count = 1;
                input_paths = argv + i;
                state = 2;
                break;
            default:;
                // Do nothing
            }
            break;
        case 2: // file inputs
            switch (get_file_type(argv[i])) {
            case 1: // regular file
                input_count++;
                break;
            default:
                i--;
                state = 9;
            }
            break;
        default:
            break;
        }
    }

    // Parse connx model
    connx_Model model;
    ret = connx_Model_init(&model);
    if (ret != 0) {
        goto done;
    }

#if DEBUG
    connx_debug("Model: %s\n", argv[1]);
#endif /* DEBUG */

    ret = run_from_file(&model, input_count, input_paths, loop);

    connx_Model_destroy(&model);

done:
    connx_destroy();

    return ret;
}
