#ifndef __CONNX_DUMP_H__
#define __CONNX_DUMP_H__

#include <connx/backend.h>

void connx_Call_dump(connx_HAL* hal, connx_Call* call, int depth);
void connx_Path_dump(connx_HAL* hal, connx_Path* path, int depth);
void connx_Backend_dump(connx_HAL* hal, connx_Backend* backend, int depth);

#endif /* __CONNX_DUMP_H__ */
