#define _POSIX_C_SOURCE 199309L
#include <inttypes.h>
#include <malloc.h>
#include <stdarg.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <connx/accel.h>
#include <connx/tensor.h>

#include <connx/hal.h>
#include <connx/types.h>

static char _model_path[128];
static FILE* _tensorin;
static FILE* _tensorout;

// Lifecycle
void connx_init() {
}

void connx_destroy() {
    if(_tensorin != NULL) {
        fclose(_tensorin);
    }

    if(_tensorout != NULL) {
        fclose(_tensorout);
    }
}

// Memory management
void* connx_alloc(uint32_t size) {
    return calloc(1, size);
}

void connx_free(void* ptr) {
    free(ptr);
}

// Model loader
int connx_set_model(const char* path) {
    snprintf(_model_path, 128, "%s", path);
    struct stat st;
    if(stat(_model_path, &st) == 0 && S_ISDIR(st.st_mode)) {
        return OK;
    } else {
        connx_error("Model not found in path: '%s'\n", _model_path);
        return RESOURCE_NOT_FOUND;
    }
}

int connx_set_tensorin(const char* path) {
    _tensorin = fopen(path, "r");
    if(_tensorin != NULL) {
        return OK;
    } else {
        connx_error("Tensor input PIPE not found in path: '%s'\n", path);
        return RESOURCE_NOT_FOUND;
    }
}

int connx_set_tensorout(const char* path) {
    _tensorout = fopen(path, "w");
    if(_tensorout != NULL) {
        return OK;
    } else {
        connx_error("Tensor output PIPE not found in path: '%s'\n", path);
        return RESOURCE_NOT_FOUND;
    }
}

void* connx_load(const char* name) {
    char path[256];
    snprintf(path, 256, "%s/%s", _model_path, name);

    FILE* file = fopen(path, "r");
    if(file == NULL) {
        fprintf(stderr, "HAL ERROR: There is no such file: '%s'\n", path);
        return NULL;
    }

    fseek(file, 0L, SEEK_END);
    size_t size = ftell(file);
    fseek(file, 0L, SEEK_SET);

    void* buf = malloc(size + 1); // including EOF
    if(buf == NULL) {
        fprintf(stderr, "HAL ERROR: Cannot allocate memory: %" PRIdPTR " bytes", size);
        fclose(file);
        return NULL;
    }

    void* p = buf;
    size_t remain = size;
    while(remain > 0) {
        int len = fread(p, 1, remain, file);
        if(len < 0) {
            fprintf(stderr, "HAL ERROR: Cannot read file: '%s'", path);
            fclose(file);
            return NULL;
        }

        p += len;
        remain -= len;
    }
    fclose(file);

    ((uint8_t*)buf)[size] = 0; // EOF

    return buf;
}

void connx_unload(__attribute__((unused)) void* buf) {
    free(buf);
}

// Tensor I/O
int32_t connx_read(void* buf, int32_t size) {
    FILE* file = _tensorin != NULL ? _tensorin : stdin;

    void* p = buf;
    size_t remain = size;
    while(remain > 0) {
        int len = fread(p, 1, remain, file);

        if(len == 0) {
            struct timespec time = {0, 10000}; // 10 us
            nanosleep(&time, &time);
        }

        if(len < 0) {
            fprintf(stderr, "HAL ERROR: Cannot read input data");
            return -1;
        }

        p += len;
        remain -= len;
    }

    return size;
}

int32_t connx_write(void* buf, int32_t size) {
    FILE* file = _tensorout != NULL ? _tensorout : stdout;

    void* p = buf;
    size_t remain = size;
    while(remain > 0) {
        int len = fwrite(p, 1, remain, file);
        if(len < 0) {
            fprintf(stderr, "HAL ERROR: Cannot read input data");
            return -1;
        }

        p += len;
        remain -= len;
    }

    fflush(file);

    return size;
}

// Lock
void connx_Lock_init(connx_Lock* lock) {
    pthread_mutex_init(lock, NULL);
}

void connx_Lock_destroy(connx_Lock* lock) {
    pthread_mutex_destroy(lock);
}

void connx_Lock_lock(connx_Lock* lock) {
    pthread_mutex_lock(lock);
}

void connx_Lock_unlock(connx_Lock* lock) {
    pthread_mutex_unlock(lock);
}

// Thread pool
uint32_t connx_Thread_alloc(uint32_t count, connx_Thread* threads) {
    return 0;
}

void connx_Thread_free(uint32_t count, connx_Thread* threads) {
}

void connx_Thread_join(uint32_t count, connx_Thread* threads) {
}

// error
void connx_debug(const char* format, ...) {
    va_list args;
    va_start(args, format);

    fprintf(stdout, "DEBUG: ");
    vfprintf(stdout, format, args);

    va_end(args);
}

void connx_info(const char* format, ...) {
    va_list args;
    va_start(args, format);

    fprintf(stdout, "INFO: ");
    vfprintf(stdout, format, args);

    va_end(args);
}

void connx_error(const char* format, ...) {
    va_list args;
    va_start(args, format);

    fprintf(stderr, "ERROR: ");
    vfprintf(stderr, format, args);

    va_end(args);
}

// Below 5 lines are copied from tensor.c
#define ITER_NDIM(iter) (iter)
#define ITER_START(iter) (iter + 1)
#define ITER_STOP(iter) (iter + 1 + iter[0])
#define ITER_STEP(iter) (iter + 1 + iter[0] * 2)
#define ITER_INDEX(iter) (iter + 1 + iter[0] * 3)

void connx_Iterator_dump(int32_t* iterator) {
    int32_t ndim = *ITER_NDIM(iterator);
    int32_t* start = ITER_START(iterator);
    int32_t* stop = ITER_STOP(iterator);
    int32_t* step = ITER_STEP(iterator);
    int32_t* index = ITER_INDEX(iterator);

    for(int32_t i = 0; i < ndim; i++)
        printf("%d ", index[i]);
    printf("/ ");
    for(int32_t i = 0; i < ndim; i++)
        printf("%d ", start[i]);
    printf("/ ");
    for(int32_t i = 0; i < ndim; i++)
        printf("%d ", stop[i]);
    printf("/ ");
    for(int32_t i = 0; i < ndim; i++)
        printf("%d ", step[i]);
    printf("\n");
}

void connx_Tensor_dump(connx_Tensor* tensor) {
    int32_t unit = -1;
    int32_t unit2 = -1;

    if(tensor->ndim == 1) {
        unit = 8;
    } else if(tensor->ndim >= 1) {
        unit = tensor->shape[tensor->ndim - 1];

        if(tensor->ndim >= 2) {
            unit2 = unit * tensor->shape[tensor->ndim - 2];
        }
    }

    // New line by matrix
#define NEWLINE()              \
    if((i + 1) % unit == 0)    \
        fprintf(stderr, "\n"); \
                               \
    if((i + 1) % unit2 == 0)   \
        fprintf(stderr, "\n");

    fprintf(stderr, "tensor < ");
    for(int32_t i = 0; i < tensor->ndim; i++) {
        fprintf(stderr, "%u ", tensor->shape[i]);
    }

    int32_t total = connx_Int32_product(tensor->ndim, tensor->shape);
    fprintf(stderr, "> = %u\n", total);

    switch(tensor->dtype) {
        case UINT8: {
            uint8_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRIu8 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case INT8: {
            int8_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRId8 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case UINT16: {
            uint16_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRIu16 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case INT16: {
            int16_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRId16 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case UINT32: {
            uint32_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRIu32 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case INT32: {
            int32_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRId32 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case UINT64: {
            uint64_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRIu64 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case INT64: {
            int64_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRId64 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case FLOAT16: {
            uint16_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%" PRIu16 " ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case FLOAT32: {
            float32_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%f ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case FLOAT64: {
            float64_t* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%f ", array[i]);
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case STRING:
            fprintf(stderr, "Not implemented yet\n");
            break;
        case BOOL: {
            bool* array = tensor->buffer;
            for(int32_t i = 0; i < total; i++) {
                fprintf(stderr, "%s ", array[i] ? "true" : "false");
                NEWLINE()
            }
            fprintf(stderr, "\n");
            break;
        }
        case COMPLEX64:
        case COMPLEX128:
        case UNDEFINED:
        default:
            fprintf(stderr, "Not implemented yet\n");
    }
}
