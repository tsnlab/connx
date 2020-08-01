#ifndef __CONNX_DUMP_H__
#define __CONNX_DUMP_H__

#include <connx/backend.h>

void connx_Call_dump(connx_HAL* hal, connx_Call* call);
void connx_Path_dump(connx_HAL* hal, connx_Path* path);
void connx_Backend_dump(connx_HAL* hal, connx_Backend* backend);
void connx_Tensor_dump(connx_HAL* hal, connx_Tensor* tensor);

#endif /* __CONNX_DUMP_H__ */
