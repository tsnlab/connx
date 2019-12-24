#ifndef __CONNX_H__
#define __CONNX_H__

#include <connx/onnx.h>

// connx
bool connx_init();

// exception
void connx_exception(char* format, ...);

// memory management
void* connx_alloc(size_t size) __attribute__((weak));
void connx_free(void* ptr) __attribute__((weak));

// data type
typedef enum _connx_DataType {
	connx_DataType_VOID		= 0x00,
	connx_DataType_UINT8	= (1 << 0),
	connx_DataType_UINT16	= (1 << 1),
	connx_DataType_UINT32	= (1 << 2),
	connx_DataType_UINT64	= (1 << 3),
	connx_DataType_UINT		= connx_DataType_UINT8 | connx_DataType_UINT16 | connx_DataType_UINT32 | connx_DataType_UINT64,
	connx_DataType_INT8		= (1 << 4),
	connx_DataType_INT16	= (1 << 5),
	connx_DataType_INT32	= (1 << 6),
	connx_DataType_INT64	= (1 << 7),
	connx_DataType_INT		= connx_DataType_INT8 | connx_DataType_INT16 | connx_DataType_INT32 | connx_DataType_INT64,
	connx_DataType_INTEGER	= connx_DataType_UINT | connx_DataType_INT,
	connx_DataType_INTEGER32= connx_DataType_UINT32 | connx_DataType_UINT64 | connx_DataType_INT32 | connx_DataType_INT64,
	connx_DataType_FLOAT16	= (1 << 8),
	connx_DataType_FLOAT32	= (1 << 9),
	connx_DataType_FLOAT64	= (1 << 10),
	connx_DataType_FLOAT	= connx_DataType_FLOAT16 | connx_DataType_FLOAT32 | connx_DataType_FLOAT64,
	connx_DataType_INTEGER32_FLOAT= connx_DataType_INTEGER32 | connx_DataType_FLOAT,
	connx_DataType_NUMBER	= connx_DataType_INTEGER| connx_DataType_FLOAT,
	connx_DataType_BOOL		= (1 << 11),
	connx_DataType_STRING	= (1 << 12),
	connx_DataType_ARRAY	= (1 << 13),
	connx_DataType_INT64_ARRAY	= connx_DataType_INT64 | connx_DataType_ARRAY,
	connx_DataType_FLOAT32_ARRAY= connx_DataType_FLOAT32 | connx_DataType_ARRAY,
	connx_DataType_STRING_ARRAY= connx_DataType_STRING | connx_DataType_ARRAY,
	connx_DataType_TENSOR	= (1 << 14),
	connx_DataType_TENSOR_INTEGER32_FLOAT= connx_DataType_TENSOR | connx_DataType_INTEGER32 | connx_DataType_FLOAT,
	connx_DataType_TENSOR_FLOAT= connx_DataType_TENSOR | connx_DataType_FLOAT,
	connx_DataType_TENSOR_NUMBER= connx_DataType_TENSOR | connx_DataType_NUMBER,
	connx_DataType_TENSOR_INT64= connx_DataType_TENSOR | connx_DataType_INT64,
	connx_DataType_SEQUENCE	= (1 << 15),
	connx_DataType_MAP		= (1 << 16),
} connx_DataType;

int connx_DataType_toString(connx_DataType type, int len, char* buf);
uint32_t connx_DataType_size(connx_DataType type);
connx_DataType connx_DataType_from_onnx(int32_t type);

// common header for Tensor, Sequence and Map
typedef struct _connx_Value {
	connx_DataType		type;
	char*				name;
} connx_Value;

typedef struct _connx_Tensor {
	connx_DataType		type;	// connx_DataType_TENSOR
	char*				name;

	connx_DataType		elemType;

	uint32_t 			dimension;
	uint32_t*			lengths;
	uint8_t				base[0];
} connx_Tensor;

typedef struct _connx_Sequence {
	connx_DataType		type;	// connx_DataType_SEQUENCE
	char*				name;

	connx_DataType		elemType;
	uint32_t			length;

	uint8_t				base[0];
} connx_Sequence;

typedef struct _connx_Map {
	connx_DataType		type;	// connx_DataType_MAP
	char*				name;

	connx_DataType		keyType;
	connx_DataType		valueType;
	uint32_t			length;

	void*				keys;
	void*				values;
} connx_Map;

connx_Tensor* connx_Tensor_create(connx_DataType type, uint32_t dimension, ...);
connx_Tensor* connx_Tensor_create2(connx_DataType type, uint32_t dimension, uint32_t* lengths);
connx_Tensor* connx_Tensor_create_from_onnx(Onnx__TensorProto* onnx);
connx_Tensor* connx_Tensor_create_from_file(const char* path);
connx_Tensor* connx_Tensor_clone(connx_Tensor* tensor);
bool connx_Tensor_copy(connx_Tensor* tensor, connx_Tensor* dest);
void connx_Tensor_clean(connx_Tensor* tensor);
void connx_Tensor_delete(connx_Tensor* tensor);
void connx_Tensor_dump(connx_Tensor* tensor);
void connx_Tensor_dump_compare(connx_Tensor* tensor, connx_Tensor* tensor2, double epsilon);
uint32_t connx_Tensor_total(connx_Tensor* tensor);
bool connx_Tensor_equals(connx_Tensor* tensor, connx_Tensor* tensor2);
bool connx_Tensor_isNearlyEquals(connx_Tensor* tensor, connx_Tensor* tensor2, double epsilon);
bool connx_Tensor_isShapeEquals(connx_Tensor* tensor, connx_Tensor* tensor2);
int connx_Tensor_toShapeString(connx_Tensor* tensor, int len, char* buf);

connx_Sequence* connx_Sequence_create(connx_DataType type, uint32_t length);
connx_Map* connx_Map_create(connx_DataType keyType, connx_DataType valueType, uint32_t length);

connx_Value* connx_Value_create_from_onnx(connx_Type* type);
connx_Value* connx_Value_clone(connx_Value* value);
bool connx_Value_copy(connx_Value* value, connx_Value* dest);
void connx_Value_clean(connx_Value* value);
void connx_Value_delete(connx_Value* value);

// Operations
typedef struct _connx_Operator {
	char*				name;

	uint32_t			outputCount;
	connx_DataType*		outputs;

	uint32_t			inputCount;
	connx_DataType*		inputs;

	uint32_t			attributeCount;
	char**				attributeNames;
	connx_DataType*		attributes;
	uintptr_t*			attributeValues;

	bool (*resolve)(uintptr_t* stack);
	bool (*exec)(uintptr_t* stack);
} connx_Operator;

extern uint32_t connx_operator_count;
extern connx_Operator connx_operators[];

void connx_Operator_add(const char* name, 
		uint32_t outputCount, uint32_t inputCount, uint32_t attributeCount,
		bool (*resolve)(uintptr_t* stack), bool (*exec)(uintptr_t* stack), ...);
connx_Operator* connx_Operator_get(const char* name);
void connx_Operator_dump();

uintptr_t connx_Attribute_create_float(float v);
uintptr_t connx_Attribute_create_int(int64_t v);
uintptr_t connx_Attribute_create_string(const char* v);
uintptr_t connx_Attribute_create_string_from_onnx(ProtobufCBinaryData* data);
uintptr_t connx_Attribute_create_floats(uint32_t length, float* v);
uintptr_t connx_Attribute_create_ints(uint32_t length, int64_t* v);
uintptr_t connx_Attribute_create_strings(uint32_t length, char** v);
uintptr_t connx_Attribute_create_strings_from_onnx(uint32_t length, ProtobufCBinaryData** v);
uintptr_t connx_Attribute_clone_float(uintptr_t attr);
uintptr_t connx_Attribute_clone_int(uintptr_t attr);
uintptr_t connx_Attribute_clone_string(uintptr_t attr);
uintptr_t connx_Attribute_clone_floats(uintptr_t attr);
uintptr_t connx_Attribute_clone_ints(uintptr_t attr);
uintptr_t connx_Attribute_clone_strings(uintptr_t attr);
void connx_Attribute_delete(void* attr);
uint32_t connx_Attribute_length(void* attr);
void* connx_Attribute_base(void* attr);

// Runtime
typedef struct _connx_Runtime {
	connx_Model*		model;

	uint32_t			dependencyCount;
	connx_Node**		dependencies;
	connx_Operator**	operators;

	uint32_t			variableCount;
	connx_Value**		variables;
	connx_Value**		initializers;

	uint32_t			inputCount;
	connx_ValueInfo**	inputs;

	uint32_t			outputCount;
	connx_ValueInfo**	outputs;

	uint32_t			stackCount;
	uintptr_t*			stack;
	bool				isDirty;
} connx_Runtime;

connx_Runtime* connx_Runtime_create(connx_Model* model);
void connx_Runtime_delete(connx_Runtime* runtime);
bool connx_Runtime_setVariable(connx_Runtime* runtime, connx_Value* value);
connx_Value* connx_Runtime_getVariable(connx_Runtime* runtime, const char* name);
connx_Value* connx_Runtime_run(connx_Runtime* runtime, connx_Value* input);

#endif /* __CONNX_H__ */
