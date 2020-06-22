/*
 * opset.h
 *
 *  Created on: Dec 8, 2019
 *      Author: semih
 */

#ifndef __CONNX_OPSET_H__
#define __CONNX_OPSET_H__

#include "connx.h"

struct connx_Operation {
	char*					name;

	uint32_t				inputCount;
	struct connx_Type*		inputs;

	uint32_t				outputCount;
	struct connx_Type*		outputs;

	uint32_t				attributeCount;
	struct connx_Attribute*	attributes;

	bool (*validate)(uintptr_t* stack);
	bool (*exec)(uintptr_t* stack);
};

extern struct connx_Operation* connx_operations;

void connx_operation_init();
struct connx_Operation* connx_operation(const char* name);

#endif /* __CONNX_OPSET_H__ */
