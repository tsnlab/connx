#ifndef __CONNX_BACKEND_H__
#define __CONNX_BACKEND_H__

#include <connx/connx.h>
#include <connx/operator.h>

typedef struct _connx_Attribute_Int {
	int32_t		value;
} __attribute__((packed)) connx_Attribute_Int;

typedef struct _connx_Attribute_Float {
	float		value;
} __attribute__((packed)) connx_Attribute_Float;

typedef struct _connx_Attribute_String {
	char		value[0];
} __attribute__((packed)) connx_Attribute_String;

typedef struct _connx_Attribute_Ints {
	uint32_t	length;
	int32_t		values[0];
} __attribute__((packed)) connx_Attribute_Ints;

typedef struct _connx_Attribute_Floats {
	uint32_t	length;
	float		values[0];
} __attribute__((packed)) connx_Attribute_Floats;

typedef struct _connx_Attribute_Strings {
	uint32_t	length;
	uint32_t	offsets[0];
} __attribute__((packed)) connx_Attribute_Strings;

#define CONNX_COUNTS(outputs, inputs, attrs) ((((outputs) + (inputs) + (attrs)) << 0) |	\
											  ((outputs) << 8) | \
											  ((inputs) << 16) | \
											  ((attrs) << 24))

#define CONNX_TOTAL_COUNT(counts) (((counts) >> 0) & 0xff)
#define CONNX_OUTPUT_COUNT(counts) (((counts) >> 8) & 0xff)
#define CONNX_INPUT_COUNT(counts) (((counts) >> 16) & 0xff)
#define CONNX_ATTRIBUTE_COUNT(counts) (((counts) >> 24) & 0xff)

typedef struct _connx_Call {
	connx_Operator	op;
	uint32_t		counts;
	uint32_t*		params;
} connx_Call;

connx_Call* connx_Call_create(connx_HAL* hal, connx_Operator op, uint32_t counts);
void connx_Call_delete(connx_HAL* hal, connx_Call* call);

typedef struct _connx_Path connx_Path;

struct _connx_Path {
	uint32_t		input_path_count;
	uint32_t*		input_paths;

	uint32_t		output_path_count;
	uint32_t*		output_paths;

	uint32_t		call_count;
	connx_Call**	calls;
};

connx_Path* connx_Path_create(connx_HAL* hal);
void connx_Path_delete(connx_HAL* hal, connx_Path* call);

struct _connx_Backend {
	connx_HAL*		hal;
	uint32_t		opset;

	uint32_t		path_count;
	connx_Path**	paths;

	uint32_t		initializer_count;
	uint32_t		variable_count;
	connx_Tensor**	variables;

	uint32_t*		attribute_index;
	void*			attributes;

	uint32_t		start_count;
	uint32_t*		starts;

	uint32_t		stop_count;
	uint32_t*		stops;

	uint32_t		clean_count;
	uint32_t*		cleans;
};

bool connx_Backend_has_variable(connx_Backend* backend, uint32_t id);
connx_Tensor* connx_Backend_get_variable(connx_Backend* backend, uint32_t id);
bool connx_Backend_set_variable(connx_Backend* backend, uint32_t id, connx_Tensor* tensor);
bool connx_Backend_delete_variable(connx_Backend* backend, uint32_t id);

bool connx_Backend_has_attribute(connx_Backend* backend, uint32_t id);
void* connx_Backend_get_attribute(connx_Backend* backend, uint32_t id);
bool connx_Backend_set_attribute(connx_Backend* backend, uint32_t id, void* attribute);
bool connx_Backend_delete_attribute(connx_Backend* backend, uint32_t id, void* attribute);

#endif /* __CONNX_BACKEND_H__ */
