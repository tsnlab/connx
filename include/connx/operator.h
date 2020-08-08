#ifndef __CONNX_OPERATOR_H__
#define __CONNX_OPERATOR_H__

#include <connx/connx.h>

// Operator
#define CONNX_COUNTS(outputs, inputs, attrs) ((((outputs) + (inputs) + (attrs)) << 0) |	\
											  ((outputs) << 8) | \
											  ((inputs) << 16) | \
											  ((attrs) << 24))

#define CONNX_TOTAL_COUNT(counts) (((counts) >> 0) & 0xff)
#define CONNX_OUTPUT_COUNT(counts) (((counts) >> 8) & 0xff)
#define CONNX_INPUT_COUNT(counts) (((counts) >> 16) & 0xff)
#define CONNX_ATTRIBUTE_COUNT(counts) (((counts) >> 24) & 0xff)

#define CONNX_GET_OUTPUT(idx)			(idx < CONNX_OUTPUT_COUNT(counts) ? connx_Backend_get_variable(backend, params[(idx)]) : NULL)
#define CONNX_SET_OUTPUT(idx, tensor)	connx_Backend_set_variable(backend, params[(idx)], (tensor))
#define CONNX_GET_INPUT(idx)			(idx < CONNX_INPUT_COUNT(counts) ? connx_Backend_get_variable(backend, params[CONNX_OUTPUT_COUNT(counts) + (idx)]) : NULL)
#define CONNX_GET_ATTRIBUTE(idx)		(idx < CONNX_ATTRIBUTE_COUNT(counts) ? connx_Backend_get_attribute(backend, params[CONNX_OUTPUT_COUNT(counts) + CONNX_INPUT_COUNT(counts) + (idx)]) : NULL)

#define CONNX_GET_OUTPUT_INDEX(idx)			(idx < CONNX_OUTPUT_COUNT(counts) ? params[(idx)] : 0)
#define CONNX_GET_INPUT_INDEX(idx)			(idx < CONNX_INPUT_COUNT(counts) ? params[CONNX_OUTPUT_COUNT(counts) + (idx)] : 0)
#define CONNX_GET_ATTRIBUTE_INDEX(idx)		(idx < CONNX_ATTRIBUTE_COUNT(counts) ? params[CONNX_OUTPUT_COUNT(counts) + CONNX_INPUT_COUNT(counts) + (idx)] : 0)

#define CONNX_GET_OUTPUT_OFFSET(idx)		(idx < CONNX_OUTPUT_COUNT(counts) ? backend->attribute_index[params[(idx)]] : 0)
#define CONNX_GET_INPUT_OFFSET(idx)			(idx < CONNX_INPUT_COUNT(counts) ? backend->attribute_index[params[CONNX_OUTPUT_COUNT(counts) + (idx)]] : 0)
#define CONNX_GET_ATTRIBUTE_OFFSET(idx)		(idx < CONNX_ATTRIBUTE_COUNT(counts) ? backend->attribute_index[params[CONNX_OUTPUT_COUNT(counts) + CONNX_INPUT_COUNT(counts) + (idx)]] : 0)

typedef struct _connx_AttributeInt {
	int32_t		value;
} connx_AttributeInt;

typedef struct _connx_AttributeFloat {
	float		value;
} connx_AttributeFloat;

typedef struct _connx_AttributeString {
	char		value[0];
} connx_AttributeString;

typedef struct _connx_AttributeInts {
	uint32_t	length;
	int32_t		values[0];
} connx_AttributeInts;

typedef struct _connx_AttributeFloats {
	uint32_t	length;
	float		values[0];
} connx_AttributeFloats;

typedef struct _connx_AttributeStrings {
	uint32_t	length;
	uint32_t	offsets[0];
} connx_AttributeStrings;

typedef bool (*connx_Operator)(connx_Backend* backend, uint32_t counts, uint32_t* params);

extern char* connx_operator_names[];
extern connx_Operator connx_operators[];

#endif /* __CONNX_OPERATOR_H__ */
