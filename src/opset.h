#ifndef __CONNX_OPSET_H__
#define __CONNX_OPSET_H__

#include <stdint.h>

#define __unit(dimension, lengths) ({			\
	uint32_t unit = 1;							\
	for(uint32_t i = 1; i < dimension; i++) {	\
		unit *= lengths[i];						\
	}											\
												\
	unit;										\
})

#define INIT(array, type)	\
	uint32_t array##_unit = __unit(array##_dimension, array##_lengths) * connx_DataType_size(type);	\
	uint32_t array##_length = array##_lengths[0];

#define BASE(array, idx) (array + array##_unit * idx)

#define FOR(array0, array1, array2)	\
	for(uint32_t array0##_idx = 0, array1##_idx = 0, array2##_idx = 0;	\
			array0##_idx < array0##_length;								\
			array0##_idx++,		\
			array1##_idx = (array1##_idx + 1) % array1##_length,		\
			array2##_idx = (array2##_idx + 1) % array2##_length)

#endif /* __CONNX_OPSET_H__ */
