#ifndef __CONNX_OPERATOR_H__
#define __CONNX_OPERATOR_H__

#include <connx/connx.h>

// Operator
typedef bool (*connx_Operator)(connx_Backend* backend, uint32_t counts, uint32_t* params);

extern char* connx_operator_names[];
extern connx_Operator connx_operators[];

#endif /* __CONNX_OPERATOR_H__ */
