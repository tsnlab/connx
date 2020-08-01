#include <stdio.h>
#include <inttypes.h>
#include <connx/dump.h>

#define MESSAGE_SIZE 256
static char _message[MESSAGE_SIZE];

const static char* DATA_TYPE_NAME[] = {
	"void", "float32", "uint8", "int8",
	"uint16", "int16", "int32", "int64",
	"string", "bool", "float16", "float64",
	"uint32", "uint64",
};

static char* find_op_name(connx_Operator op) {
	for(int i = 0; connx_operators[i] != NULL; i++) {
		if(connx_operators[i] == op) {
			return connx_operator_names[i];
		}
	}

	return "NOP";
}

void connx_Call_dump(connx_HAL* hal, connx_Call* call) {
	hal->debug(hal, "call");
}

void connx_Path_dump(connx_HAL* hal, connx_Path* path) {
	hal->debug(hal, "path");
}

void connx_Backend_dump(connx_HAL* hal, connx_Backend* backend) {
	hal->debug(hal, "backend");
}

void connx_Tensor_dump(connx_HAL* hal, connx_Tensor* tensor) {
	const char* type_name = DATA_TYPE_NAME[tensor->type];

	uint32_t total = 1;
	for(uint32_t i = 0; i < tensor->dimension; i++) {
		total *= tensor->lengths[i];
	}

	char* m = _message;
	int size = MESSAGE_SIZE;

	int len = snprintf(m, size, "<%s> [ ", type_name);
	m += len; size -= len; if(size < 0) goto done;

	for(uint32_t i = 0; i < tensor->dimension; i++) {
		len = snprintf(m, size, "%u", tensor->lengths[i]);
		m += len; size -= len; if(size < 0) goto done;

		if(i + 1 < tensor->dimension) {
			len = snprintf(m, size, ", ");
			m += len; size -= len; if(size < 0) goto done;
		}
	}

	len = snprintf(m, size, " ] {");
	m += len; size -= len; if(size < 0) goto done;

	hal->debug(hal, _message);

	m = _message;
	size = MESSAGE_SIZE;

	len = snprintf(m, size, "\t");
	m += len; size -= len; if(size < 0) goto done;

#define MAX_ELEMENT_COUNT	32
#define NEWLINE_COUNT		8
#define NEWLINE											\
	if(i != 0 && i % NEWLINE_COUNT == 0) {				\
		hal->debug(hal, _message);						\
														\
		m = _message;									\
		size = MESSAGE_SIZE;							\
														\
		len = snprintf(m, size, "\t");					\
		m += len; size -= len; if(size < 0) goto done;	\
	}

	switch(tensor->type) {
		case connx_VOID:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "<void> ");
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
		case connx_FLOAT32:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "%f ", ((float*)tensor->base)[i]);
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
		case connx_UINT8:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "%" PRIu32 " ", ((uint8_t*)tensor->base)[i]);
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
		case connx_INT8:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "%" PRId32 " ", ((int8_t*)tensor->base)[i]);
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
		case connx_UINT16:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "%" PRIu32 " ", ((uint16_t*)tensor->base)[i]);
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
		case connx_INT16:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "%" PRId32 " ", ((int16_t*)tensor->base)[i]);
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
		case connx_INT32:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "%" PRId32 " ", ((int32_t*)tensor->base)[i]);
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
		case connx_INT64:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "%" PRId64 " ", ((int64_t*)tensor->base)[i]);
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
		case connx_STRING:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "%s ", ((char**)tensor->base)[i]);
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
		case connx_BOOL:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "%s ", ((bool*)tensor->base)[i] ? "true" : "false");
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
		case connx_FLOAT16:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "%" PRIx32 " ", ((uint16_t*)tensor->base)[i]);
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
		case connx_FLOAT64:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "%f ", ((double*)tensor->base)[i]);
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
		case connx_UINT32:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "%" PRIu32 " ", ((uint32_t*)tensor->base)[i]);
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
		case connx_UINT64:
			for(uint32_t i = 0; i < total && i < MAX_ELEMENT_COUNT; i++) {
				NEWLINE
				len = snprintf(m, size, "%" PRIu64 " ", ((uint64_t*)tensor->base)[i]);
				m += len; size -= len; if(size < 0) goto done;
			}
			break;
	}

	if(_message != m) {
		hal->debug(hal, _message);
	}

	if(total > MAX_ELEMENT_COUNT) {
		snprintf(_message, MESSAGE_SIZE, "... (%u remains)", total - MAX_ELEMENT_COUNT);
		hal->debug(hal, _message);
	}

	hal->debug(hal, "}");
	return;

done:
	hal->debug(hal, _message);
}
