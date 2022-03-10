#ifndef __CONNX_HAL_H__
#define __CONNX_HAL_H__

#include <stdint.h>

#include <connx/hal_common.h>

int hal_set_model(const char* path);
int hal_set_tensorin(const char* path);
int hal_set_tensorout(const char* path);
int32_t hal_read(void* buf, int32_t size);
int32_t hal_write(void* buf, int32_t size);


#endif // __CONNX_HAL_H__
