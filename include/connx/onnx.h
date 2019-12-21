#ifndef __ONNX_H__
#define __ONNX_H__

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <onnx/onnx.proto3.pb-c.h>

typedef struct _Onnx__AttributeProto onnx_AttributeProto;
typedef struct _Onnx__ValueInfoProto onnx_ValueInfoProto;
typedef struct _Onnx__NodeProto onnx_NodeProto;
typedef struct _Onnx__ModelProto onnx_ModelProto;
typedef struct _Onnx__StringStringEntryProto onnx_StringStringEntryProto;
typedef struct _Onnx__TensorAnnotation onnx_TensorAnnotation;
typedef struct _Onnx__GraphProto onnx_GraphProto;
typedef struct _Onnx__TensorProto onnx_TensorProto;
typedef struct _Onnx__TensorProto__Segment onnx_TensorProto_Segment;
typedef struct _Onnx__SparseTensorProto onnx_SparseTensorProto;
typedef struct _Onnx__TensorShapeProto onnx_TensorShapeProto;
typedef struct _Onnx__TensorShapeProto__Dimension onnx_TensorShapeProto_Dimension;
typedef struct _Onnx__TypeProto onnx_TypeProto;
typedef struct _Onnx__TypeProto__Tensor onnx_TypeProto_Tensor;
typedef struct _Onnx__TypeProto__Sequence onnx_TypeProto_Sequence;
typedef struct _Onnx__TypeProto__Map onnx_TypeProto_Map;
typedef struct _Onnx__OperatorSetIdProto onnx_OperatorSetIdProto;

onnx_ModelProto* onnx_Model_create_from_file(const char* path);
void onnx_ModelProto_delete(onnx_ModelProto* onnx);

void onnx_TensorProto_delete(onnx_TensorProto* onnx);

void onnx_Attribute_dump(onnx_AttributeProto* attribute);
void onnx_ValueInfo_dump(onnx_ValueInfoProto* valueInfo);
void onnx_Node_dump(onnx_NodeProto* node);
void onnx_Model_dump(onnx_ModelProto* model);
void onnx_Graph_dump(onnx_GraphProto* graph);
void onnx_Tensor_dump(onnx_TensorProto* tensor);
void onnx_SparseTensor_dump(onnx_SparseTensorProto* sparse);
void onnx_TensorType_dump(onnx_TypeProto_Tensor* type);
void onnx_SequenceType_dump(onnx_TypeProto_Sequence* type);
void onnx_MapType_dump(onnx_TypeProto_Map* type);
void onnx_Type_dump(onnx_TypeProto* type);
void onnx_DataType_dump(int32_t type);

#endif /* __ONNX_H__ */
