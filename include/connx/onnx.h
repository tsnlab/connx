#ifndef __ONNX_H__
#define __ONNX_H__

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <onnx/onnx.proto3.pb-c.h>

typedef struct _Onnx__AttributeProto connx_Attribute;
typedef struct _Onnx__ValueInfoProto connx_ValueInfo;
typedef struct _Onnx__NodeProto connx_Node;
typedef struct _Onnx__ModelProto connx_Model;
typedef struct _Onnx__StringStringEntryProto connx_StringStringEntry;
typedef struct _Onnx__TensorAnnotation connx_TensorAnnotation;
typedef struct _Onnx__GraphProto connx_Graph;
typedef struct _Onnx__TensorProto__Segment connx_Tensor_Segment;
typedef struct _Onnx__SparseTensorProto connx_SparseTensor;
typedef struct _Onnx__TensorShapeProto connx_TensorShape;
typedef struct _Onnx__TensorShapeProto__Dimension connx_TensorShape_Dimension;
typedef struct _Onnx__TypeProto connx_Type;
typedef struct _Onnx__TypeProto__Tensor connx_Type_Tensor;
typedef struct _Onnx__TypeProto__Sequence connx_Type_Sequence;
typedef struct _Onnx__TypeProto__Map connx_Type_Map;
typedef struct _Onnx__OperatorSetIdProto connx_OperatorSetId;

connx_Model* connx_Model_create_from_file(const char* path);
void connx_Model_delete(connx_Model* onnx);
void connx_Model_dump(connx_Model* model);

#endif /* __ONNX_H__ */
