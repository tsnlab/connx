#ifndef __CONNX_HAL_H__
#define __CONNX_HAL_H__

#include <stdbool.h>
#include <stdint.h>
#ifdef __linux__
#include <pthread.h>
#endif /* __linux__ */

#define CONNX_ALIGNMENT 16 // Data alignment
#define CONNX_ALIGN(offset) (((offset) + CONNX_ALIGNMENT - 1) & ~(CONNX_ALIGNMENT - 1))

// Lifecycle
void connx_init();
void connx_destroy();

// Memory management
void* connx_alloc(uint32_t size);
void connx_free(void* ptr);

// Model loader
#ifdef __linux__
int connx_set_model(const char* path);
int connx_set_tensorin(const char* path);
int connx_set_tensorout(const char* path);
#endif
void* connx_load(const char* name);
void connx_unload(void* buf);

// Tensor I/O
int32_t connx_read(void* buf, int32_t size);
int32_t connx_write(void* buf, int32_t size);

// Lock
#ifdef __linux__
typedef pthread_mutex_t connx_Lock;
#endif /* __linux__ */

void connx_Lock_init(connx_Lock* lock);
void connx_Lock_destroy(connx_Lock* lock);

void connx_Lock_lock(connx_Lock* lock);
void connx_Lock_unlock(connx_Lock* lock);

// Thread pool
#ifdef __linux__
typedef pthread_t connx_Thread;
#endif /* __linux__ */

uint32_t connx_Thread_alloc(uint32_t count, connx_Thread* threads);
void connx_Thread_free(uint32_t count, connx_Thread* threads);
void connx_Thread_join(uint32_t count, connx_Thread* threads);

// error
void connx_debug(const char* format, ...);
void connx_info(const char* format, ...);
void connx_error(const char* format, ...);

#endif /* __CONNX_HAL_H__ */
