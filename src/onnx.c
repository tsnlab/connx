#include "onnx/onnx.proto3.pb-c.h"
#include <stdio.h>
#include <string.h>
#include <connx/connx.h>

static void* _default_alloc(__attribute__((unused)) void* ctx, size_t size) {
	return connx_alloc(size);
}

static void _default_free(__attribute__((unused)) void* ctx, void* ptr) {
	connx_free(ptr);
}

static ProtobufCAllocator _default_allocator = {
	.alloc = _default_alloc,
	.free = _default_free,
	.allocator_data = NULL
};

connx_Model* connx_Model_create_from_file(const char* path) {
	FILE* file = fopen(path, "r");
	if(file == NULL) {
		connx_exception("'%s' is not found.", path);
		return NULL;
	}

	fseek(file, 0L, SEEK_END);
	long len = ftell(file);
	fseek(file, 0L, SEEK_SET);

	uint8_t* buf = connx_alloc(len);
	if(buf == NULL) {
		connx_exception("Out of memory: cannot allocate %ld bytes", len);
		return NULL;
	}

	size_t len2 = fread(buf, 1, len, file);
	fclose(file);

	if((long)len2 != len) {
		connx_free(buf);
		connx_exception("Cannot fully read ONNX file, expected: %d != read: %d", len, len2);
		return NULL;
	}

	connx_Model* onnx = onnx__model_proto__unpack(&_default_allocator, len, buf);
	connx_free(buf);
	if(onnx == NULL) {
		connx_exception("Illegal ONNX format: %s", path);
		return NULL;
	}

	return onnx;
}

void connx_Model_delete(connx_Model* onnx) {
	onnx__model_proto__free_unpacked(onnx, &_default_allocator);
}

connx_Tensor* connx_Tensor_create_from_file(const char* path) {
	FILE* file = fopen(path, "r");
	if(file == NULL) {
		connx_exception("'%s' is not found.", path);
		return NULL;
	}

	fseek(file, 0L, SEEK_END);
	long len = ftell(file);

	uint8_t buf[len];
	fseek(file, 0L, SEEK_SET);
	size_t len2 = fread(buf, 1, len, file);
	fclose(file);

	if((long)len2 != len) {
		connx_exception("Cannot fully read ONNX file, expected: %d != read: %d", len, len2);
		return NULL;
	}

	Onnx__TensorProto* onnx = onnx__tensor_proto__unpack(&_default_allocator, len, buf);
	if(onnx == NULL) {
		connx_exception("'%s' is not an onnx file.", path);
		return NULL;
	}

	connx_Tensor* tensor = connx_Tensor_create_from_onnx(onnx);
	onnx__tensor_proto__free_unpacked(onnx, &_default_allocator);

	return tensor;
}

static int depth = 0;

static void tab() {
	for(int i = 0; i < depth; i++)
		fprintf(stdout, "\t");
}

static void onnx_Attribute_dump(connx_Attribute* attribute);
static void onnx_ValueInfo_dump(connx_ValueInfo* valueInfo);
static void onnx_Node_dump(connx_Node* node);
static void onnx_Graph_dump(connx_Graph* graph);
static void onnx_Tensor_dump(Onnx__TensorProto* tensor);
static void onnx_SparseTensor_dump(connx_SparseTensor* sparse);
static void onnx_TensorType_dump(connx_Type_Tensor* type);
static void onnx_SequenceType_dump(connx_Type_Sequence* type);
static void onnx_MapType_dump(connx_Type_Map* type);
static void onnx_Type_dump(connx_Type* type);
static void onnx_DataType_dump(int32_t type);

void onnx_Attribute_dump(connx_Attribute* attribute) {
	tab(); fprintf(stdout, "Attribute: %s\n", attribute->name);
	depth++;

	tab(); fprintf(stdout, "ref_attr_name=%s\n",attribute->ref_attr_name);

	if(attribute->doc_string != NULL) {
		tab(); fprintf(stdout, "doc_string=%s\n", attribute->doc_string);
	}

	switch(attribute->type) {
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT:
			tab(); fprintf(stdout, "float: %f\n", attribute->f);
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT:
			tab(); fprintf(stdout, "int: %ld\n", attribute->i);
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING:
			{
				char buf[attribute->s.len + 1];
				memcpy(buf, attribute->s.data, attribute->s.len);
				buf[attribute->s.len] = 0;

				tab(); fprintf(stdout, "string: %s\n", buf);
			}
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR:
			onnx_Tensor_dump(attribute->t);
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH:
			onnx_Graph_dump(attribute->g);
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR:
			onnx_SparseTensor_dump(attribute->sparse_tensor);
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS:
			tab(); fprintf(stdout, "float: [");
			for(size_t i = 0; i < attribute->n_floats; i++) {
				fprintf(stdout, "%f", attribute->floats[i]);
				if(i + 1 < attribute->n_floats)
					fprintf(stdout, ", ");
			}
			fprintf(stdout, "]\n");
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS:
			tab(); fprintf(stdout, "int: [");
			for(size_t i = 0; i < attribute->n_ints; i++) {
				fprintf(stdout, "%ld", attribute->ints[i]);
				if(i + 1 < attribute->n_ints)
					fprintf(stdout, ", ");
			}
			fprintf(stdout, "]\n");
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS:
			tab(); fprintf(stdout, "string: [");
			for(size_t i = 0; i < attribute->n_strings; i++) {
				char buf[attribute->strings[i].len + 1];
				memcpy(buf, attribute->strings[i].data, attribute->strings[i].len);
				buf[attribute->strings[i].len] = 0;

				fprintf(stdout, "%s", buf);
				if(i + 1 < attribute->n_strings)
					fprintf(stdout, ", ");
			}
			fprintf(stdout, "]\n");
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSORS:
			tab(); fprintf(stdout, "tensor: [");
			for(size_t i = 0; i < attribute->n_tensors; i++) {
				onnx_Tensor_dump(attribute->tensors[i]);
			}
			fprintf(stdout, "]\n");
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPHS:
			tab(); fprintf(stdout, "graph: [");
			for(size_t i = 0; i < attribute->n_graphs; i++) {
				onnx_Graph_dump(attribute->graphs[i]);
			}
			fprintf(stdout, "]\n");
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSORS:
			tab(); fprintf(stdout, "sparseTensor: [");
			for(size_t i = 0; i < attribute->n_sparse_tensors; i++) {
				onnx_SparseTensor_dump(attribute->sparse_tensors[i]);
			}
			fprintf(stdout, "]\n");
			break;
		case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__UNDEFINED:
		default:
			tab(); fprintf(stdout, "undefined");
	}

	depth--;
}

void onnx_ValueInfo_dump(connx_ValueInfo* valueInfo) {
	tab(); fprintf(stdout, "ValueInfo: %s, ", valueInfo->name);
	onnx_Type_dump(valueInfo->type);

	if(valueInfo->doc_string != NULL) {
		fprintf(stdout, ", doc_string: %s\n", valueInfo->doc_string);
	} else {
		fprintf(stdout, "\n");
	}
}

void onnx_Node_dump(connx_Node* node)  {
	tab(); fprintf(stdout, "%s: %s\n", node->op_type, node->name);

	depth++;

	tab(); fprintf(stdout, "input: ");
	for(size_t i = 0; i < node->n_input; i++) {
		fprintf(stdout, "%s", node->input[i]);

		if(i + 1 < node->n_input)
			fprintf(stdout, ", ");
	}
	fprintf(stdout, "\n");

	tab(); fprintf(stdout, "output: ");
	for(size_t i = 0; i < node->n_output; i++) {
		fprintf(stdout, "%s", node->output[i]);

		if(i + 1 < node->n_output)
			fprintf(stdout, ", ");
	}
	fprintf(stdout, "\n");

	tab(); fprintf(stdout, "domain: %s\n", node->domain);

	tab(); fprintf(stdout, "attribute: %ld\n", node->n_attribute);

	depth++;
	for(size_t i = 0; i < node->n_attribute; i++) {
		onnx_Attribute_dump(node->attribute[i]);
	}
	depth--;

	if(node->doc_string != NULL) {
		tab(); fprintf(stdout, "doc_string: %s\n", node->doc_string);
	}

	depth--;
}

void connx_Model_dump(connx_Model* model) {
	tab(); fprintf(stdout, "ir_version: %ld\n", model->ir_version);
	tab(); fprintf(stdout, "opset_import\n");

	for(size_t i = 0; i < model->n_opset_import; i++) {
		tab(); fprintf(stdout, "\tdomain: %s, ver: %ld\n", model->opset_import[i]->domain, model->opset_import[i]->version);
	}

	tab(); fprintf(stdout, "producer_name: %s\n", model->producer_name);
	tab(); fprintf(stdout, "producer_version: %s\n", model->producer_version);
	tab(); fprintf(stdout, "domain: %s\n", model->domain);
	tab(); fprintf(stdout, "model_version: %ld\n", model->model_version);

	if(model->doc_string != NULL) {
		tab(); fprintf(stdout, "doc_string: %s\n", model->doc_string);
	}

	for(size_t i = 0; i < model->n_metadata_props; i++) {
		tab(); fprintf(stdout, "\tkey: %s, value: %s\n", model->metadata_props[i]->key, model->metadata_props[i]->value);
	}
	onnx_Graph_dump(model->graph);
}

void onnx_Graph_dump(connx_Graph* graph) {
	tab(); fprintf(stdout, "Graph: %s\n", graph->name);

	depth++;

	tab(); fprintf(stdout, "initializer: %ld\n", graph->n_initializer);
	depth++;
	for(size_t i = 0; i < graph->n_initializer; i++) {
		onnx_Tensor_dump(graph->initializer[i]);

		if(i + 1 < graph->n_initializer)
			fprintf(stdout, "\n");
	}
	depth--;

	tab(); fprintf(stdout, "sparse_initializer: %ld\n", graph->n_sparse_initializer);
	for(size_t i = 0; i < graph->n_sparse_initializer; i++) {
		tab(); fprintf(stdout, "[%ld]\n", i);
		depth++;
		onnx_SparseTensor_dump(graph->sparse_initializer[i]);
		depth--;
	}

	if(graph->doc_string != NULL) {
		tab(); fprintf(stdout, "doc_string: %s\n", graph->doc_string);
	}

	tab(); fprintf(stdout, "input\n");
	depth++;
	for(size_t i = 0; i < graph->n_input; i++) {
		onnx_ValueInfo_dump(graph->input[i]);
	}
	depth--;

	tab(); fprintf(stdout, "output\n");
	depth++;
	for(size_t i = 0; i < graph->n_output; i++) {
		onnx_ValueInfo_dump(graph->output[i]);
	}
	depth--;

	tab(); fprintf(stdout, "value_info\n");
	depth++;
	for(size_t i = 0; i < graph->n_value_info; i++) {
		onnx_ValueInfo_dump(graph->value_info[i]);
	}
	depth--;

	tab(); fprintf(stdout, "quantization_annotation\n");
	depth++;
	for(size_t i = 0; i < graph->n_quantization_annotation; i++) {
		Onnx__TensorAnnotation* anno = graph->quantization_annotation[i];
		fprintf(stdout, "tensor_name: %s, qualt_parameter_tensor_names: ", anno->tensor_name);
		for(size_t j = 0; j < anno->n_quant_parameter_tensor_names; j++) {
			fprintf(stdout, "(key: %s, value: %s)", anno->quant_parameter_tensor_names[j]->key, anno->quant_parameter_tensor_names[j]->value);
			if(j + 1 < anno->n_quant_parameter_tensor_names)
				fprintf(stdout, ", ");
		}
		fprintf(stdout, "\n");
	}
	depth--;

	tab(); fprintf(stdout, "node\n");
	depth++;
	for(size_t i = 0; i < graph->n_node; i++) {
		onnx_Node_dump(graph->node[i]);
	}
	depth--;

	depth--;
}

void onnx_Tensor_dump(Onnx__TensorProto* tensor) {
#define MAX_DATA_COUNT 32

	tab(); fprintf(stdout, "Tensor: %s\n", tensor->name);

	depth++;

	tab(); onnx_DataType_dump(tensor->data_type);
	fprintf(stdout, "[ ");
	for(size_t i = 0; i < tensor->n_dims; i++) {
		fprintf(stdout, "%ld", tensor->dims[i]);

		if(i + 1 < tensor->n_dims)
			fprintf(stdout, ", ");
	}
	if(tensor->segment != NULL) {
		fprintf(stdout, "| %ld ~ %ld ", tensor->segment->begin, tensor->segment->end);
	}

	fprintf(stdout, " ] : ");

	size_t len = 0;
	switch(tensor->data_type) {
		case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
			len = tensor->n_float_data > MAX_DATA_COUNT ? MAX_DATA_COUNT : tensor->n_float_data;

			for(size_t i = 0; i < len; i++) {
				fprintf(stdout, "%f ", tensor->float_data[i]);
			}

			if(len < tensor->n_float_data) {
				fprintf(stdout, "... (%lu more)", tensor->n_float_data - len);
			}

			fprintf(stdout, "\n");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
		case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
			len = tensor->n_int32_data > MAX_DATA_COUNT ? MAX_DATA_COUNT : tensor->n_int32_data;

			for(size_t i = 0; i < len; i++) {
				fprintf(stdout, "%d ", tensor->int32_data[i]);
			}

			if(len < tensor->n_int32_data) {
				fprintf(stdout, "... (%lu more)", tensor->n_int32_data - len);
			}

			fprintf(stdout, "\n");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
			len = tensor->n_string_data > MAX_DATA_COUNT ? MAX_DATA_COUNT : tensor->n_string_data;

			for(size_t i = 0; i < len; i++) {
				char buf[tensor->string_data[i].len + 1];
				memcpy(buf, tensor->string_data[i].data, tensor->string_data[i].len);
				buf[tensor->string_data[i].len] = 0;

				fprintf(stdout, "%s (%lu)", tensor->string_data[i].data, tensor->string_data[i].len);
			}

			if(len < tensor->n_string_data) {
				fprintf(stdout, "... (%lu more)", tensor->n_string_data - len);
			}

			fprintf(stdout, "\n");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
			len = tensor->n_int64_data > MAX_DATA_COUNT ? MAX_DATA_COUNT : tensor->n_int64_data;

			for(size_t i = 0; i < len; i++) {
				fprintf(stdout, "%ld ", tensor->int64_data[i]);
			}

			if(len < tensor->n_int64_data) {
				fprintf(stdout, "... (%lu more)", tensor->n_int64_data - len);
			}

			fprintf(stdout, "\n");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
			len = tensor->n_double_data > MAX_DATA_COUNT ? MAX_DATA_COUNT : tensor->n_double_data;

			for(size_t i = 0; i < len; i++) {
				fprintf(stdout, "%f ", tensor->double_data[i]);
			}

			if(len < tensor->n_double_data) {
				fprintf(stdout, "... (%lu more)", tensor->n_double_data - len);
			}

			fprintf(stdout, "\n");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
			len = tensor->n_uint64_data > MAX_DATA_COUNT ? MAX_DATA_COUNT : tensor->n_uint64_data;

			for(size_t i = 0; i < len; i++) {
				fprintf(stdout, "%lu ", tensor->uint64_data[i]);
			}

			if(len < tensor->n_uint64_data) {
				fprintf(stdout, "... (%lu more)", tensor->n_uint64_data - len);
			}

			fprintf(stdout, "\n");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
		default:
			; // Do nothing
	}

	if(tensor->doc_string != NULL) {
		tab(); fprintf(stdout, "doc_string: %s\n", tensor->doc_string);
	}

	tab(); fprintf(stdout, "raw_data: %ld", tensor->raw_data.len);
	tab();
	len = tensor->raw_data.len > 64 ? 64 : tensor->raw_data.len;
	for(size_t i = 0; i < len; i++) {
		fprintf(stdout, "%u ", tensor->raw_data.data[i]);
	}

	if(len < tensor->raw_data.len) {
		fprintf(stdout, "... (%lu more)", tensor->raw_data.len - len);
	}

	fprintf(stdout, "\n");

	// TODO: externel data
	// TODO: data location

	depth--;
}

void onnx_SparseTensor_dump(connx_SparseTensor* sparse) {
	tab(); fprintf(stdout, "values: ");
	onnx_Tensor_dump(sparse->values);
	tab(); fprintf(stdout, "indices: ");
	onnx_Tensor_dump(sparse->indices);
	tab(); fprintf(stdout, "dims: [%ld] ", sparse->n_dims);
	for(size_t i = 0; i < sparse->n_dims; i++) {
		fprintf(stdout, "%ld ", sparse->dims[i]);
	}
	fprintf(stdout, "\n");
}

void onnx_TensorType_dump(connx_Type_Tensor* type) {
	fprintf(stdout, "Tensor<");
	onnx_DataType_dump(type->elem_type);
	fprintf(stdout, ">[ ");

	Onnx__TensorShapeProto* shape = type->shape;
	for(size_t i = 0; i < shape->n_dim; i++) {
		switch(shape->dim[i]->value_case) {
		case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
			fprintf(stdout, "%ld", shape->dim[i]->dim_value);
			break;
		case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
			fprintf(stdout, "%s", shape->dim[i]->dim_param);
			break;
		case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE__NOT_SET:
		default:
			fprintf(stdout, "not set");
		}

		if(i + 1 < shape->n_dim)
			fprintf(stdout, ", ");
	}
	fprintf(stdout, " ]");
}

void onnx_SequenceType_dump(connx_Type_Sequence* type) {
	fprintf(stdout, "sequence<");
	onnx_Type_dump(type->elem_type);
	fprintf(stdout, ">");
}

void onnx_MapType_dump(connx_Type_Map* type) {
	fprintf(stdout, "map<");
	onnx_DataType_dump(type->key_type);
	fprintf(stdout, ", ");
	onnx_Type_dump(type->value_type);
	fprintf(stdout, ">");
}

void onnx_Type_dump(connx_Type* type) {
	switch(type->value_case) {
		case ONNX__TYPE_PROTO__VALUE__NOT_SET:
			fprintf(stdout, "N/A");
			break;
		case ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE:
			onnx_TensorType_dump(type->tensor_type);
			break;
		case ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE:
			onnx_SequenceType_dump(type->sequence_type);
			break;
		case ONNX__TYPE_PROTO__VALUE_MAP_TYPE:
			onnx_MapType_dump(type->map_type);
			break;
		default:
			;
	}
}

void onnx_DataType_dump(int32_t type) {
	switch(type) {
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
			fprintf(stdout, "float");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
			fprintf(stdout, "uint8");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
			fprintf(stdout, "int8");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
			fprintf(stdout, "uint16");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
			fprintf(stdout, "int16");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
			fprintf(stdout, "int32");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
			fprintf(stdout, "int64");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
			fprintf(stdout, "string");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
			fprintf(stdout, "bool");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
			fprintf(stdout, "float16");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
			fprintf(stdout, "double");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
			fprintf(stdout, "uint32");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
			fprintf(stdout, "uint64");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
			fprintf(stdout, "complex64");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
			fprintf(stdout, "complex128");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
			fprintf(stdout, "bfloat16");
			break;
		case ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED:
		default:
			fprintf(stdout, "undefined");
			break;
	}
}
