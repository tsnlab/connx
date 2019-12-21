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

//
///**
// * tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double)
// */
//struct connx_Tensor* connx_Add(struct connx_Tensor* A, struct connx_Tensor* B);
//
///**
// *  tensor(float16), tensor(float), tensor(double)
// */
//struct connx_Tensor* connx_Relu(struct connx_Tensor* X);
//
///**
// *  @param data tensor(*)
// *  @param shape tensor(int64)
// *  @return data's type
// */
//struct connx_Tensor* connx_Reshape(struct connx_Tensor* data, struct connx_Tensor* shape);
//
///**
// * tensor(float16), tensor(float), tensor(double), tensor(uint32), tensor(uint64), tensor(int32), tensor(int64)
// */
//struct connx_Tensor* connx_Matmul(struct connx_Tensor* A, struct connx_Tensor* B);
//
///**
// * tensor(float16), tensor(float), tensor(double), tensor(uint32), tensor(uint64), tensor(int32), tensor(int64)
// */
//struct connx_Tensor* connx_MaxPool(struct connx_Tensor* X, int32_t ceil_mode, uint32_t dilations_count, int32_t* dilations, uint32_t kernel_shape_count, int32_t* kernel_shape, uint32_t pads_count, int32_t* pads, int32_t storage_order, uint32_t strides_count, int32_t* strides);
//
///**
// * tensor(float16), tensor(float), tensor(double)
// */
//struct connx_Tensor* connx_Conv(struct connx_Tensor* X, struct connx_Tensor* W, const char* auto_pad, uint32_t dilations_count, int32_t* dilations, int32_t group, uint32_t kernel_shape_count, int32_t* kernel_shape, uint32_t pads_count, int32_t* pads, uint32_t strides_count, int32_t* strides);

#endif /* __CONNX_OPSET_H__ */
