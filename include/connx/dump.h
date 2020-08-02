#ifndef __CONNX_DUMP_H__
#define __CONNX_DUMP_H__

#include <connx/backend.h>
#include <connx/operator.h>

void connx_Call_dump(connx_HAL* hal, connx_Call* call);
void connx_Path_dump(connx_HAL* hal, connx_Path* path);
void connx_Backend_dump(connx_HAL* hal, connx_Backend* backend);
void connx_Tensor_dump(connx_HAL* hal, connx_Tensor* tensor);

void connx_AttributeInt_dump(connx_HAL* hal, const char* title, connx_AttributeInt* attr);
void connx_AttributeFloat_dump(connx_HAL* hal, const char* title, connx_AttributeFloat* attr);
void connx_AttributeString_dump(connx_HAL* hal, const char* title, connx_AttributeString* attr);
void connx_AttributeInts_dump(connx_HAL* hal, const char* title, connx_AttributeInts* attr);
void connx_AttributeFloats_dump(connx_HAL* hal, const char* title, connx_AttributeFloats* attr);
void connx_AttributeStrings_dump(connx_HAL* hal, const char* title, connx_AttributeStrings* attr);

#endif /* __CONNX_DUMP_H__ */
