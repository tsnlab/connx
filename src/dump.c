#include <stdio.h>
#include <inttypes.h>
#include <connx/dump.h>

extern bool opset_delete(connx_Backend* backend, uint32_t counts, uint32_t* params);

static const char* find_op_name(connx_Operator op) {
	if(op == opset_delete)
		return "delete";

	for(int i = 0; connx_operators[i] != NULL; i++) {
		if(connx_operators[i] == op) {
			return connx_operator_names[i];
		}
	}

	return "nop";
}

void connx_Call_dump(connx_HAL* hal, connx_Call* call) {
	const char* op_name = find_op_name(call->op);

	hal->debug(hal, "%s %u %u %u  ", op_name, 
			CONNX_OUTPUT_COUNT(call->counts),
			CONNX_INPUT_COUNT(call->counts),
			CONNX_ATTRIBUTE_COUNT(call->counts));

	uint32_t total = CONNX_TOTAL_COUNT(call->counts);
	for(uint32_t i = 0; i < total; i++) {
		hal->debug(hal, "%u ", call->params[i]);
	}

	hal->debug(hal, "\n");
}

void connx_Path_dump(connx_HAL* hal, connx_Path* path) {
	hal->debug(hal, "path");
}

void connx_Backend_dump(connx_HAL* hal, connx_Backend* backend) {
	hal->debug(hal, "backend");
}

void connx_Tensor_dump(connx_HAL* hal, connx_Tensor* tensor) {
	const char* type_name = connx_DataType_name(tensor->type);

	uint32_t total = 1;
	for(uint32_t i = 0; i < tensor->dimension; i++) {
		total *= tensor->lengths[i];
	}

	hal->debug(hal, "<%s> [ ", type_name);

	for(uint32_t i = 0; i < tensor->dimension; i++) {
		hal->debug(hal, "%u", tensor->lengths[i]);

		if(i + 1 < tensor->dimension) {
			hal->debug(hal, ", ");
		}
	}

	hal->debug(hal, " ] {\n");

	hal->debug(hal, "\t");

#define MAX_ELEMENT_COUNT	64
#define NEWLINE_COUNT		8
#define NEWLINE()										\
	if(i != 0 && i % NEWLINE_COUNT == 0) {				\
		hal->debug(hal, "\n");							\
		hal->debug(hal, "\t");							\
	}

	switch(tensor->type) {
		case connx_VOID:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "<void> ");
			}
			break;
		case connx_FLOAT32:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "%f ", ((float*)tensor->base)[i]);
			}
			break;
		case connx_UINT8:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "%" PRIu32 " ", ((uint8_t*)tensor->base)[i]);
			}
			break;
		case connx_INT8:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "%" PRId32 " ", ((int8_t*)tensor->base)[i]);
			}
			break;
		case connx_UINT16:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "%" PRIu32 " ", ((uint16_t*)tensor->base)[i]);
			}
			break;
		case connx_INT16:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "%" PRId32 " ", ((int16_t*)tensor->base)[i]);
			}
			break;
		case connx_INT32:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "%" PRId32 " ", ((int32_t*)tensor->base)[i]);
			}
			break;
		case connx_INT64:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "%" PRId64 " ", ((int64_t*)tensor->base)[i]);
			}
			break;
		case connx_STRING:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "%s ", ((char**)tensor->base)[i]);
			}
			break;
		case connx_BOOL:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "%s ", ((bool*)tensor->base)[i] ? "true" : "false");
			}
			break;
		case connx_FLOAT16:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "%" PRIx32 " ", ((uint16_t*)tensor->base)[i]);
			}
			break;
		case connx_FLOAT64:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "%f ", ((double*)tensor->base)[i]);
			}
			break;
		case connx_UINT32:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "%" PRIu32 " ", ((uint32_t*)tensor->base)[i]);
			}
			break;
		case connx_UINT64:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE();
				hal->debug(hal, "%" PRIu64 " ", ((uint64_t*)tensor->base)[i]);
			}
			break;
	}

	if(total > MAX_ELEMENT_COUNT) {
		hal->debug(hal, "... (%u remains)\n", total - MAX_ELEMENT_COUNT);
	} else {
		hal->debug(hal, "\n", total - MAX_ELEMENT_COUNT);
	}

	hal->debug(hal, "}\n");
}

void connx_AttributeInt_dump(connx_HAL* hal, connx_AttributeInt* attr) {
	hal->debug(hal, "%" PRId32, attr->value);
}

void connx_AttributeFloat_dump(connx_HAL* hal, connx_AttributeFloat* attr) {
	hal->debug(hal, "%f", attr->value);
}

void connx_AttributeString_dump(connx_HAL* hal, connx_AttributeString* attr) {
	hal->debug(hal, "%s", attr->value);
}

void connx_AttributeInts_dump(connx_HAL* hal, connx_AttributeInts* attr) {
	for(uint32_t i = 0; i < attr->length; i++) {
		hal->debug(hal, "%" PRId32 " ", attr->values[i]);
	}
}

void connx_AttributeFloats_dump(connx_HAL* hal, connx_AttributeFloats* attr) {
	for(uint32_t i = 0; i < attr->length; i++) {
		hal->debug(hal, "%f ", attr->values[i]);
	}
}

void connx_AttributeStrings_dump(connx_HAL* hal, connx_AttributeStrings* attr) {
	for(uint32_t i = 0; i < attr->length; i++) {
		hal->debug(hal, "\"%s\" ", (char*)attr + attr->offsets[i]);
	}
}

