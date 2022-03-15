#include <ff.h>
#include <stdio.h>

#include <connx/connx.h>
#include <connx/hal.h>

/* FreeRTOS includes. */
#include "FreeRTOS.h"
#include "queue.h"
#include "task.h"
#include "timers.h"

/* Xilinx includes. */
#include "xil_printf.h"
#include "xparameters.h"

static FIL* open_tensorin(const char* path) {
    TCHAR* root = "0:/";
    FIL* tensorin;
    FATFS fatfs;
    FRESULT res;

    res = f_mount(&fatfs, root, 0);

    if (res != FR_OK) {
        connx_error("%s %s\n", __func__, "f_mount fails\n");
        return NULL;
    }

    res = f_open(tensorin, path, FA_READ);
    if (res != FR_OK) {
        connx_error("%s %s\n", __func__, "f_open fails\n");
        return NULL;
    }
    return tensorin;
}

static FIL* open_tensorout(const char* path) {
    TCHAR* root = "0:/";
    FIL* tensorout;
    FATFS fatfs;
    FRESULT res;

    res = f_mount(&fatfs, root, 0);

    if (res != FR_OK) {
        connx_error("%s %s\n", __func__, "f_mount fails\n");
        return NULL;
    }

    res = f_open(tensorout, path, FA_CREATE_NEW | FA_WRITE);

    if (res != FR_OK) {
        connx_error("%s %s\n", __func__, "f_open fails\n");
        return NULL;
    }

    return tensorout;
}

static int close_tensor(FIL* fp) {
    return f_close(fp);
}

static int32_t file_read(FIL* fp, void* buf, int32_t size) {
    size_t remain = size;
    void* p = buf;
    FRESULT res;
    UINT br;

    while (remain > 0) {
        res = f_read(fp, p, remain, &br);
        if (res != FR_OK) {
            connx_error("%s %s\n", __func__, "Cannot read input data\n");
            f_close(fp);
            return -1;
        }
        p += br;
        remain -= br;
    }

    return size;
}

static int32_t file_write(FIL* fp, void* buf, int32_t size) {
    void* p = buf;
    FRESULT res;
    UINT br;

    size_t remain = size;
    while (remain > 0) {
        res = f_write(fp, p, remain, &br);
        if (res != FR_OK) {
            connx_error("%s %s\n", __func__, "f_write fails\n");
            f_close(fp);
            return -1;
        }
        p += br;
        remain -= br;
    }

    return size;
}

static int read_tensor(FIL* fp, connx_Tensor** tensor) {
    int32_t dtype;

    if (file_read(fp, &dtype, sizeof(int32_t)) != (int32_t)sizeof(int32_t)) {
        connx_error("%s %s\n", __func__, "Cannot read input data type from Tensor I/O module.\n");

        return CONNX_IO_ERROR;
    }

    int32_t ndim;

    if (file_read(fp, &ndim, sizeof(int32_t)) != (int32_t)sizeof(int32_t)) {
        connx_error("%s %s\n", __func__, "Cannot read input ndim from Tensor I/O module.\n");

        return CONNX_IO_ERROR;
    }

    int32_t shape[ndim];

    if (file_read(fp, &shape, sizeof(int32_t) * ndim) != (int32_t)(sizeof(int32_t) * ndim)) {
        connx_error("%s %s\n", __func__, "Cannot read input shape from Tensor I/O module.\n");

        return CONNX_IO_ERROR;
    }

    *tensor = connx_Tensor_alloc(dtype, ndim, shape);
    if (*tensor == NULL) {
        connx_error("%s %s\n", __func__, "Out of memory\n");

        return CONNX_NOT_ENOUGH_MEMORY;
    }

    int32_t len = file_read(fp, (*tensor)->buffer, (*tensor)->size);

    if (len != (int32_t)(*tensor)->size) {
        connx_error("%s %s\n", __func__, "Cannot read input data from Tensor I/O moudle.\n");

        return CONNX_IO_ERROR;
    }
    return CONNX_OK;
}

static int run_from_file(connx_Model* model, int input_count) {
    FIL* tensorin;
    int ret;

    // Read input tensors
    connx_Tensor* inputs[input_count];

    for (int i = 0; i < input_count; i++) {
        // Open file
        tensorin = open_tensorin("input_0.dat");
        if (tensorin == NULL) {
            return CONNX_RESOURCE_NOT_FOUND;
        }

        connx_Tensor* tensor;
        ret = read_tensor(tensorin, &tensor);

        if (ret != CONNX_OK) {
            connx_error("%s %s\n", __func__, "Cannot read tensor: %s\n", "input_0.dat");
            return ret;
        }

        inputs[i] = tensor;

        // Close file
        close_tensor(tensorin);
    }

    // Run model
    uint32_t output_count = 16;
    connx_Tensor* outputs[output_count];

    ret = connx_Model_run(model, input_count, inputs, &output_count, outputs);
    if (ret != CONNX_OK) {
        connx_error("%s %s\n", __func__, "Cannot run model\n");
        return ret;
    }

    for (uint32_t j = 0; j < output_count; j++) {
        connx_Tensor_dump(outputs[j]);
    }

    return CONNX_OK;
}

int main(void) {
    int32_t ret = CONNX_OK;
    int input_count = 1;
    connx_Model model;

    ret = connx_Model_init(&model);
    if (ret != CONNX_OK) {
        connx_error("%s %s\n", __func__, "connx_Model_init fails\n");
        return -1;
    }

    ret = run_from_file(&model, input_count);
    if (ret != CONNX_OK)
        connx_error("%s %s\n", __func__, "run_from_file fails\n");

    connx_Model_destroy(&model);

    return 0;
}
