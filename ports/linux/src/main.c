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
#include <connx/connx.h>
#include <malloc.h>

int connx_set_model(const char* path);
int connx_set_tensorin(const char* path);
int connx_set_tensorout(const char* path);

void test9() {
    int32_t shape[3] = {3, 3, 3};
    connx_Tensor* X = connx_Tensor_alloc(CONNX_UINT8, 3, shape);
    uint8_t* x_array = X->buffer;
    for (int i = 0; i < 3 * 3 * 3; i++) {
        x_array[i] = i;
    }

    connx_Tensor* mask = connx_Tensor_alloc(CONNX_UINT8, 3, shape);
    connx_Tensor* mask_ret = NULL;
    uint8_t* mask_array = mask->buffer;
    for (int i = 0; i < 3 * 3 * 3; i++) {
        mask_array[i] = 0;
    }

    connx_Slice* slices = (connx_Slice*)malloc(3 * sizeof(connx_Slice));

    fprintf(stderr, " TEST CASE 9 set by slice : start > end 인데 step이 음수  \n");
    connx_Slice_init(&slices[0], 0, 3, 2, 0);
    connx_Slice_init(&slices[1], 0, 3, 1, 0);
    connx_Slice_init(&slices[2], 0, 3, 1, 0);

    mask_ret = connx_Tensor_get_by_slice(mask, slices);
    if (mask_ret != NULL) {
        connx_Tensor_set_by_slice(X, slices, mask_ret);
        connx_Tensor_dump(mask_ret);
        connx_Tensor_dump(X);
    }
}

void test_slice_3d() {
    int32_t shape[5] = {3, 3, 5, 5, 5};
    connx_Tensor* X = connx_Tensor_alloc(CONNX_UINT8, 5, shape);
    uint8_t* x_array = X->buffer;
    for (int i = 0; i < 3 * 3 * 5 * 5 * 5; i++) {
        x_array[i] = 9;
    }

    connx_Tensor* mask = connx_Tensor_alloc(CONNX_UINT8, 5, shape);
    connx_Tensor* mask_ret = NULL;
    uint8_t* mask_array = mask->buffer;
    for (int i = 0; i < 3 * 3 * 5 * 5 * 5; i++) {
	mask_array[i] = 0;
    }
 
    connx_Slice* slices = (connx_Slice*)malloc(5 * sizeof(connx_Slice));
    /*
    // 테스트 1 : 기본 레인지 테스트
    fprintf(stderr, " TEST CASE 1: 배치 채널 step 2로 가져오기 :\n");
    connx_Slice_init(&slices[0], 1, 3, 2, 3);
    connx_Slice_init(&slices[1], 1, 3, 2, 3);
    connx_Slice_init(&slices[2], 0, 5, 1, 5);
    connx_Slice_init(&slices[3], 0, 5, 1, 5);
    connx_Slice_init(&slices[4], 0, 5, 1, 5);
    ret = connx_Tensor_get_by_slice(X, slices);
    connx_Tensor_dump(ret);

    // 테스트 2 : 행 1:5 열 1:5 뽑되 스텝 2로 행열 모두 주기
    fprintf(stderr, " TEST CASE 2\n");
    connx_Slice_init(&slices[0], 0, 1, 2, 1);
    connx_Slice_init(&slices[1], 0, 1, 2, 1);
    connx_Slice_init(&slices[2], 0, 1, 2, 1);
    connx_Slice_init(&slices[3], 0, 5, 2, 5);
    connx_Slice_init(&slices[4], 0, 5, 2, 5);
    ret = connx_Tensor_get_by_slice(X, slices);
    connx_Tensor_dump(ret);

    // 테스트 3 : set by slice 기본 테스트, 
    fprintf(stderr, " TEST CASE 3 : set by slice 테스트  \n");
    connx_Slice_init(&slices[0], 0, 1, 2, 1);
    connx_Slice_init(&slices[1], 0, 1, 2, 1);
    connx_Slice_init(&slices[2], 0, 1, 3, 1);
    connx_Slice_init(&slices[3], 1, 5, 2, 5);
    connx_Slice_init(&slices[4], 1, 5, 1, 5);

    mask_ret = connx_Tensor_get_by_slice(mask, slices);
    ret = connx_Tensor_get_by_slice(X, slices);

    connx_Tensor_set_by_slice(X, slices, mask_ret);
    connx_Tensor_dump(ret);
    */
    // 테스트 4 : shape이 넘어가는 step
    fprintf(stderr, "TEST CASE 4 set by slice : step이 shape을 넘어갈 때 \n");
    connx_Slice_init(&slices[0], 0, 1, 6, 0);
    connx_Slice_init(&slices[1], 0, 1, 2, 0);
    connx_Slice_init(&slices[2], 0, 5, 3, 0);
    connx_Slice_init(&slices[3], 1, 5, 2, 1);
    connx_Slice_init(&slices[4], 1, 5, 2, 1);

    mask_ret = connx_Tensor_get_by_slice(mask, slices);
    if (mask_ret == NULL) {
        fprintf(stderr, "MASK NULL! TEST SUCCESS\n");
    } else {
        fprintf(stderr, "MASK NOT NULL, TEST FAILED\n");
    }

    // 테스트 5 : 음수인 step
    fprintf(stderr, "TEST CASE 5 set by slice : step이 음수 일 때 \n");
    connx_Slice_init(&slices[0], 0, 1, -2, 0);
    connx_Slice_init(&slices[1], 0, 1, 2, 0);
    connx_Slice_init(&slices[2], 0, 5, 3, 0);
    connx_Slice_init(&slices[3], 1, 5, 2, 1);
    connx_Slice_init(&slices[4], 1, 5, 2, 1);

    mask_ret = connx_Tensor_get_by_slice(mask, slices);

    if (mask_ret == NULL) {
        fprintf(stderr, "MASK NULL! TEST SUCCESS\n");
    } else {
        connx_Tensor_set_by_slice(X, slices, mask_ret);
        connx_Tensor_dump(mask_ret);
        connx_Tensor_dump(X);
    }

    // 테스트 6 : start, end가 음수
    fprintf(stderr, " TEST CASE 6 set by slice : start end 가 음수일 때  \n");
    connx_Slice_init(&slices[0], -1, -2, 2, 0);
    connx_Slice_init(&slices[1], 0, 1, 2, 0);
    connx_Slice_init(&slices[2], 0, 5, 3, 0);
    connx_Slice_init(&slices[3], 1, 5, 2, 1);
    connx_Slice_init(&slices[4], 1, 5, 2, 1);

    if (mask_ret == NULL) {
        fprintf(stderr, "MASK NULL! TEST SUCCESS\n");
    } else {
        connx_Tensor_set_by_slice(X, slices, mask_ret);
        connx_Tensor_dump(mask_ret);
        connx_Tensor_dump(X);
    }

    // 테스트 7 : start, end가 0일 때
    fprintf(stderr, " TEST CASE 7 set by slice : start, end가 0일 때   \n");
    connx_Slice_init(&slices[0], 0, 0, -2, 0);
    connx_Slice_init(&slices[1], 0, 1, 2, 0);
    connx_Slice_init(&slices[2], 0, 5, 3, 0);
    connx_Slice_init(&slices[3], 1, 5, 2, 1);
    connx_Slice_init(&slices[4], 1, 5, 2, 1);
 
    mask_ret = connx_Tensor_get_by_slice(mask, slices);
    if (mask_ret == NULL) {
        fprintf(stderr, "MASK NULL! TEST SUCCESS\n");
    } else {
        connx_Tensor_set_by_slice(X, slices, mask_ret);
        connx_Tensor_dump(mask_ret);
        connx_Tensor_dump(X);
    }

    // 테스트 8 : step이 0일 때
    fprintf(stderr, " TEST CASE 8 set by slice : step이 0일 때   \n");
    connx_Slice_init(&slices[0], 0, 1, 0, 0);
    connx_Slice_init(&slices[1], 0, 1, 2, 0);
    connx_Slice_init(&slices[2], 0, 5, 3, 0);
    connx_Slice_init(&slices[3], 1, 5, 2, 1);
    connx_Slice_init(&slices[4], 1, 5, 2, 1);
    
    mask_ret = connx_Tensor_get_by_slice(mask, slices);
    if (mask_ret == NULL) {
        fprintf(stderr, "MASK NULL! TEST SUCCESS\n");
    } else {
        connx_Tensor_set_by_slice(X, slices, mask_ret);
        connx_Tensor_dump(mask_ret);
        connx_Tensor_dump(X);
    }

    // 테스트 9 : start > end 인데 step < 0
    test9();

    fprintf(stderr, "\nSLICE TEST OVER\n\n");
}

void test_slice() {
    fprintf(stderr, "3d TEST start \n");
    test_slice_3d();
}


int main(int argc, char** argv) {
    if(argc < 2) {
        connx_info("Usage: connx [connx model path] [[tensor in pipe] tensor out pipe]]\n");
        return 0;
    }

    int32_t ret;

    ret = connx_set_model(argv[1]);
    if(ret != 0) {
        return ret;
    }

    if(argc > 3) {
        ret = connx_set_tensorin(argv[2]);
        if(ret != 0) {
            return ret;
        }

        ret = connx_set_tensorout(argv[3]);
        if(ret != 0) {
            return ret;
        }
    }

    // Parse connx model
    connx_Model model;
    ret = connx_Model_init(&model);
    if(ret != 0) {
        return ret;
    }

    // loop: input -> inference -> output
    // If input_count is -1 then exit the loop
    while(true) {
        // Read input count from HAL
        uint32_t input_count;
        if(connx_read(&input_count, sizeof(uint32_t)) != (int32_t)sizeof(uint32_t)) {
            connx_error("Cannot read input count from Tensor I/O module.\n");

            ret = -CONNX_IO_ERROR;
            connx_write(&ret, sizeof(int32_t));

            return CONNX_IO_ERROR;
        }

        if(input_count == (uint32_t)-1) {
            break;
        }

        // Read input data from HAL
        connx_Tensor* inputs[input_count];
        for(uint32_t i = 0; i < input_count; i++) {
            int32_t dtype;
            if(connx_read(&dtype, sizeof(int32_t)) != (int32_t)sizeof(int32_t)) {
                connx_error("Cannot read input data type from Tensor I/O module.\n");

                ret = -CONNX_IO_ERROR;
                connx_write(&ret, sizeof(int32_t));

                return CONNX_IO_ERROR;
            }

            int32_t ndim;
            if(connx_read(&ndim, sizeof(int32_t)) != (int32_t)sizeof(int32_t)) {
                connx_error("Cannot read input ndim from Tensor I/O module.\n");

                ret = -CONNX_IO_ERROR;
                connx_write(&ret, sizeof(int32_t));

                return CONNX_IO_ERROR;
            }

            int32_t shape[ndim];
            if(connx_read(&shape, sizeof(int32_t) * ndim) != (int32_t)(sizeof(int32_t) * ndim)) {
                connx_error("Cannot read input shape from Tensor I/O module.\n");

                ret = -CONNX_IO_ERROR;
                connx_write(&ret, sizeof(int32_t));

                return CONNX_IO_ERROR;
            }

            connx_Tensor* input = inputs[i] = connx_Tensor_alloc(dtype, ndim, shape);
            if(input == NULL) {
                connx_error("Out of memory\n");

                ret = -CONNX_NOT_ENOUGH_MEMORY;
                connx_write(&ret, sizeof(int32_t));

                return CONNX_NOT_ENOUGH_MEMORY;
            }

            int32_t len = connx_read(input->buffer, input->size);
            if(len != (int32_t)input->size) {
                connx_error("Cannot read input data from Tensor I/O moudle.\n");

                ret = -CONNX_IO_ERROR;
                connx_write(&ret, sizeof(int32_t));

                return CONNX_IO_ERROR;
            }
        }

	test_slice();

        // Run model
        uint32_t output_count = 16;
        connx_Tensor* outputs[output_count];

        ret = connx_Model_run(&model, input_count, inputs, &output_count, outputs);
        if(ret != CONNX_OK) {
            int32_t ret2 = -ret;
            connx_write(&ret2, sizeof(int32_t));

            return ret;
        }

        // Write outputs
        // Write output count
        if(connx_write(&output_count, sizeof(uint32_t)) != (int32_t)sizeof(uint32_t)) {
            connx_error("Cannot write output data to Tensor I/O moudle.\n");

            ret = -CONNX_IO_ERROR;
            connx_write(&ret, sizeof(int32_t));

            return CONNX_IO_ERROR;
        }

        for(uint32_t i = 0; i < output_count; i++) {
            connx_Tensor* output = outputs[i];

            int32_t dtype = output->dtype;
            if(connx_write(&dtype, sizeof(int32_t)) != (int32_t)sizeof(int32_t)) {
                connx_error("Cannot write output data type to Tensor I/O module.\n");

                ret = -CONNX_IO_ERROR;
                connx_write(&ret, sizeof(int32_t));

                return CONNX_IO_ERROR;
            }

            if(connx_write(&output->ndim, sizeof(int32_t)) != (int32_t)sizeof(int32_t)) {
                connx_error("Cannot write output ndim to Tensor I/O module.\n");

                ret = -CONNX_IO_ERROR;
                connx_write(&ret, sizeof(int32_t));

                return CONNX_IO_ERROR;
            }

            if(connx_write(output->shape, sizeof(int32_t) * output->ndim) != (int32_t)(sizeof(int32_t) * output->ndim)) {
                connx_error("Cannot write output shape to Tensor I/O module.\n");

                ret = -CONNX_IO_ERROR;
                connx_write(&ret, sizeof(int32_t));

                return CONNX_IO_ERROR;
            }

            if(connx_write(output->buffer, output->size) != (int32_t)output->size) {
                connx_error("Cannot write output data to Tensor I/O module.\n");

                ret = -CONNX_IO_ERROR;
                connx_write(&ret, sizeof(int32_t));

                return CONNX_IO_ERROR;
            }
        }

        /*
        for(uint32_t i = 0; i < output_count; i++) {
            printf("***** output[%u]\n", i);
            connx_Tensor_dump(outputs[i]);
        }
        */

        for(uint32_t i = 0; i < output_count; i++) {
            connx_Tensor_unref(outputs[i]);
        }
    }

    connx_Model_destroy(&model);

    return 0;
}
