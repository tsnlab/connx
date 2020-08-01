#ifndef __CONNX_CONNX_H__
#define __CONNX_CONNX_H__

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// data structure
// DataType follows ONNX's datatype enumeration number
typedef enum _connx_DataType {
	connx_VOID		= 0,
	connx_FLOAT32	= 1,
	connx_UINT8		= 2,
	connx_INT8		= 3,
	connx_UINT16	= 4,
	connx_INT16		= 5,
	connx_INT32		= 6,
	connx_INT64		= 7,
	connx_STRING	= 8,
	connx_BOOL		= 9,
	connx_FLOAT16	= 10,
	connx_FLOAT64	= 11,
	connx_UINT32	= 12,
	connx_UINT64	= 13,
} connx_DataType;

// Tensor
typedef struct _connx_Tensor {
	connx_DataType		type;

	uint32_t 			dimension;
	uint32_t*			lengths;

	uint8_t				base[0] __attribute__((aligned(16)));		// Align 16 bytes for vector operation
} connx_Tensor;

typedef struct _connx_HAL connx_HAL;

connx_Tensor* connx_Tensor_create(connx_HAL* hal, connx_DataType type, uint32_t dimension, uint32_t* lengths);
void connx_Tensor_delete(connx_HAL* hal, connx_Tensor* tensor);

// Backend
typedef struct _connx_Backend connx_Backend;

connx_Backend* connx_Backend_create(connx_HAL* hal);
void connx_Backend_delete(connx_Backend* backend);

connx_Tensor** connx_Backend_run(connx_Backend* backend, connx_Tensor** inputs);

// Hardware Abstraction Layer
typedef void* connx_Thread;

struct _connx_HAL {
	// Memory management
	void* (*alloc)(connx_HAL* hal, size_t size);
	void (*free)(connx_HAL* hal, void* ptr);

	// Model loader
	void* (*load)(connx_HAL* hal, const char* name);
	void (*unload)(connx_HAL* hal, void* buf);

	// Thread pool
	connx_Thread* (*alloc_threads)(connx_HAL* hal, uint32_t count);
	void (*free_thread)(connx_HAL* hal, connx_Thread* thread);
	connx_Thread* (*join)(connx_HAL* hal, connx_Thread* thread);

	// error
	void (*info)(connx_HAL* hal, const char* msg);
	void (*error)(connx_HAL* hal, const char* msg);

	uint8_t priv[0];
};

#endif /* __CONNX_CONNX_H__ */
