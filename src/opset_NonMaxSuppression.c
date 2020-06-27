#include <inttypes.h>
#include <stdlib.h>
#include <float.h>
#include <connx/connx.h>

static bool NonMaxSuppression_resolve(uintptr_t* stack) {
	connx_Tensor* selected_indices = (void*)stack[1];	// batch_index, 3(num_batches, class_index, box_index)
	connx_Tensor* boxes = (void*)stack[2];				// num_batches, spactial_dimension, 4(coords)
	connx_Tensor* scores = (void*)stack[3];				// num_batches, num_classes, spatial_dimension
	connx_Tensor* max_output_boxes_per_class = (void*)stack[4];	// scalar
	connx_Tensor* iou_threshold = (void*)stack[5];				// scalar
	connx_Tensor* score_threshold = (void*)stack[6];			// scalar
	int64_t* center_point_box = (void*)stack[7];				// 0 or 1

	if(selected_indices->dimension != 2) {
		connx_exception("Illegal selected_indices dimension: %" PRIu32 ", expected: %" PRIu32, selected_indices->dimension, 2);
		return false;
	}

	if(selected_indices->lengths[1] != 3) {
		connx_exception("Illegal selected_indices length[1]: %" PRIu32 ", expected: %" PRIu32, selected_indices->lengths[1], 3);
		return false;
	}

	// Check selected_indices.num_selected_indices

	if(boxes->dimension != 3) {
		connx_exception("Illegal boxes dimension: %" PRIu32 ", expected: %" PRIu32, boxes->dimension, 3);
		return false;
	}

	if(boxes->lengths[2] != 4) {
		connx_exception("Illegal boxes length[2]: %" PRIu32 ", expected: %" PRIu32, boxes->lengths[2], 4);
		return false;
	}

	if(scores->dimension != 3) {
		connx_exception("Illegal scores dimension: %" PRIu32 ", expected: %" PRIu32, scores->dimension, 3);
		return false;
	}

	if(scores->lengths[0] != boxes->lengths[0]) {
		connx_exception("Illegal scores length[0]: %" PRIu32 ", expected: %" PRIu32, scores->lengths[0], boxes->lengths[0]);
		return false;
	}

	if(scores->lengths[2] != boxes->lengths[1]) {
		connx_exception("Illegal scores length[2]: %" PRIu32 ", expected: %" PRIu32, scores->lengths[2], boxes->lengths[1]);
		return false;
	}

	if(scores->elemType != boxes->elemType) {
		connx_exception("Illegal scores elemType: %" PRIu32 ", expected: %" PRIu32, scores->elemType, boxes->elemType);
		return false;
	}

	if(max_output_boxes_per_class == NULL) {
		max_output_boxes_per_class = connx_Tensor_create(connx_DataType_INT64, 1, 1);
		*(int64_t*)max_output_boxes_per_class->base = 0;
		stack[4] = (uintptr_t)max_output_boxes_per_class;
	}

	if(max_output_boxes_per_class->dimension != 1) {
		connx_exception("Illegal max_output_boxes_per_class dimension: %" PRIu32 ", expected: %" PRIu32, max_output_boxes_per_class->dimension, 1);
		return false;
	}

	if(iou_threshold == NULL) {
		iou_threshold = connx_Tensor_create(boxes->elemType, 1, 1);
		switch(boxes->elemType) {
			case connx_DataType_FLOAT32:
				*(float*)iou_threshold->base = 0.0;
				break;
			case connx_DataType_FLOAT64:
				*(double*)iou_threshold->base = 0.0;
				break;
			default:
				abort();
		}
		stack[5] = (uintptr_t)iou_threshold;
	}

	if(iou_threshold->dimension != 1) {
		connx_exception("Illegal iou_threshold dimension: %" PRIu32 ", expected: %" PRIu32, iou_threshold->dimension, 1);
		return false;
	}

	if(iou_threshold->elemType != boxes->elemType) {
		connx_exception("Illegal iou_threshold elemType: %" PRIu32 ", expected: %" PRIu32, iou_threshold->elemType, boxes->elemType);
		return false;
	}

	if(score_threshold == NULL) {
		score_threshold = connx_Tensor_create(boxes->elemType, 1, 1);
		switch(boxes->elemType) {
			case connx_DataType_FLOAT32:
				*(float*)score_threshold->base = -FLT_MAX;
				break;
			case connx_DataType_FLOAT64:
				*(double*)score_threshold->base = -DBL_MAX;
				break;
			default:
				abort();
		}
		stack[6] = (uintptr_t)score_threshold;
	}

	if(score_threshold->dimension != 1) {
		connx_exception("Illegal score_threshold dimension: %" PRIu32 ", expected: %" PRIu32, score_threshold->dimension, 1);
		return false;
	}

	if(score_threshold->elemType != boxes->elemType) {
		connx_exception("Illegal score_threshold elemType: %" PRIu32 ", expected: %" PRIu32, score_threshold->elemType, boxes->elemType);
		return false;
	}

	if(*center_point_box != 0) { // convert x, y, w, h to y1, x1, y2, x2
		*center_point_box = 0;

		switch(boxes->elemType) {
			case connx_DataType_FLOAT32:
				{
					float* base = (float*)boxes->base;

					uint32_t count = boxes->lengths[0] * boxes->lengths[1];
					for(uint32_t i = 0; i < count; i++, base += 4) {
						float half_width = base[2] / 2;
						float half_height = base[3] / 2;
						float x1 = base[0] - half_width;
						float x2 = base[0] + half_width;
						float y1 = base[1] - half_height;
						float y2 = base[1] + half_height;

						base[0] = y1;
						base[1] = x1;
						base[2] = y2;
						base[3] = x2;
					}
				}
				break;
			case connx_DataType_FLOAT64:
				{
					double* base = (double*)boxes->base;

					uint32_t count = boxes->lengths[0] * boxes->lengths[1];
					for(uint32_t i = 0; i < count; i++, base += 4) {
						double half_width = base[2] / 2;
						double half_height = base[3] / 2;
						double x1 = base[0] - half_width;
						double x2 = base[0] + half_width;
						double y1 = base[1] - half_height;
						double y2 = base[1] + half_height;

						base[0] = y1;
						base[1] = x1;
						base[2] = y2;
						base[3] = x2;
					}
				}
				break;
			default:
				abort();
		}
	} else {	// convert flipped coordinates
		switch(boxes->elemType) {
			case connx_DataType_FLOAT32:
				{
					float* base = (float*)boxes->base;

					uint32_t count = boxes->lengths[0] * boxes->lengths[1];
					for(uint32_t i = 0; i < count; i++, base += 4) {
						if(base[0] > base[2]) {
							float tmp = base[0];
							base[0] = base[2];
							base[2] = tmp;
						}

						if(base[1] > base[3]) {
							float tmp = base[1];
							base[1] = base[3];
							base[3] = tmp;
						}
					}
				}
				break;
			case connx_DataType_FLOAT64:
				{
					double* base = (double*)boxes->base;

					uint32_t count = boxes->lengths[0] * boxes->lengths[1];
					for(uint32_t i = 0; i < count; i++, base += 4) {
						if(base[0] > base[2]) {
							double tmp = base[0];
							base[0] = base[2];
							base[2] = tmp;
						}

						if(base[1] > base[3]) {
							double tmp = base[1];
							base[1] = base[3];
							base[3] = tmp;
						}
					}
				}
				break;
			default:
				abort();
		}
	}

	return true;
}

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

/**
 * @param A y1, x1, y2, x2
 * @param B y1, x1, y2, x2
 * @return iou
 */
static float iou_float32(float* A, float* B) {
	float A_y1 = A[0];
	float A_x1 = A[1];
	float A_y2 = A[2];
	float A_x2 = A[3];

	float B_y1 = B[0];
	float B_x1 = B[1];
	float B_y2 = B[2];
	float B_x2 = B[3];

	float x1 = MAX(A_x1, B_x1);
	float x2 = MIN(A_x2, B_x2);
	float y1 = MAX(A_y1, B_y1);
	float y2 = MIN(A_y2, B_y2);

	if(x2 < x1 || y2 < y1) {
		return 0;
	} else {
		float intersection = (x2 - x1) * (y2 - y1);
		float A = (A_x2 - A_x1) * (A_y2 - A_y1);
		float B = (B_x2 - B_x1) * (B_y2 - B_y1);

		return intersection / (A + B - intersection);
	}
}

/**
 * @param A y1, x1, y2, x2
 * @param B y1, x1, y2, x2
 * @return iou
 */
static double iou_float64(double* A, double* B) {
	double A_y1 = A[0];
	double A_x1 = A[1];
	double A_y2 = A[2];
	double A_x2 = A[3];

	double B_y1 = B[0];
	double B_x1 = B[1];
	double B_y2 = B[2];
	double B_x2 = B[3];

	double x1 = MAX(A_x1, B_x1);
	double x2 = MIN(A_x2, B_x2);
	double y1 = MAX(A_y1, B_y1);
	double y2 = MIN(A_y2, B_y2);

	if(x2 < x1 || y2 < y1) {
		return 0;
	} else {
		double intersection = (x2 - x1) * (y2 - y1);
		double A = (A_x2 - A_x1) * (A_y2 - A_y1);
		double B = (B_x2 - B_x1) * (B_y2 - B_y1);

		return intersection / (A + B - intersection);
	}
}

#define BOXES(idx0, idx1) &boxes_base[(idx0) * boxes_units[0] + (idx1)* boxes_units[1]]
#define SCORES(idx0, idx1, idx2) scores_base[(idx0) * scores_units[0] + (idx1)* scores_units[1] + (idx2) * scores_units[2]]

// Ref: https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.g137784ab86_4_4784
static void process_float32(connx_Tensor* selected_indices, connx_Tensor* boxes, connx_Tensor* scores,
		int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold) {
	int64_t* selected_indices_base = (int64_t*)selected_indices->base;

	float* boxes_base = (float*)boxes->base;
	uint32_t boxes_units[3];
	boxes_units[2] = 1;
	boxes_units[1] = boxes->lengths[2] * boxes_units[2];
	boxes_units[0] = boxes->lengths[1] * boxes_units[1];

	float* scores_base = (float*)scores->base;
	uint32_t scores_units[3];
	scores_units[2] = 1;
	scores_units[1] = scores->lengths[2] * scores_units[2];
	scores_units[0] = scores->lengths[1] * scores_units[1];

	uint32_t batch_count = scores->lengths[0];
	uint32_t class_count = scores->lengths[1];
	uint32_t coord_count = scores->lengths[2];
	int32_t index[batch_count][class_count][coord_count];

	// Make index
	for(uint32_t batch = 0; batch < batch_count; batch++) {
		for(uint32_t klass = 0; klass < class_count; klass++) {
			for(uint32_t i = 0; i < coord_count; i++) {
				float score = -FLT_MAX;
				int32_t idx = -1;

				for(uint32_t coord = 0; coord < coord_count; coord++) {
					float s = SCORES(batch, klass, coord);
					
					if(s > score || idx < 0) {
						for(uint32_t j = 0; j < i; j++) {
							if(index[batch][klass][j] == (int32_t)coord) {
								goto found;
							}
						}
						score = s;
						idx = coord;
found:
						;
					}
				}

				index[batch][klass][i] = idx;
			}
		}
	}

	// non max suppression
	for(uint32_t batch = 0; batch < batch_count; batch++) {
		for(uint32_t klass = 0; klass < class_count; klass++) {
			for(uint32_t coord = 0; coord < coord_count; coord++) {
				int32_t bbox_max = index[batch][klass][coord];
				if(bbox_max < 0)
					continue;

				for(uint32_t i = coord + 1; i < coord_count; i++) {
					int32_t bbox_cur = index[batch][klass][i];
					if(bbox_cur < 0)
						continue;

					float score_cur = SCORES(batch, klass, bbox_cur);
					if(score_cur < score_threshold)
						continue;

					float iou = iou_float32(BOXES(batch, bbox_max), BOXES(batch, bbox_cur));
					if(iou < iou_threshold)
						continue;

					index[batch][klass][i] = -1;
				}
			}
		}
	}

	// output
	uint32_t selected_idx = 0;
	for(uint32_t batch = 0; batch < batch_count; batch++) {
		for(uint32_t klass = 0; klass < class_count; klass++) {
			uint32_t klass_count = 0;

			for(uint32_t coord = 0; coord < coord_count; coord++) {
				int32_t idx = index[batch][klass][coord];
				if(idx < 0)
					continue;

				if(klass_count++ >= max_output_boxes_per_class || selected_idx++ >= selected_indices->lengths[0])
					break;

				*selected_indices_base++ = batch;
				*selected_indices_base++ = klass;
				*selected_indices_base++ = idx;
			}
		}
	}
}

static void process_float64(connx_Tensor* selected_indices, connx_Tensor* boxes, connx_Tensor* scores,
		int64_t max_output_boxes_per_class, double iou_threshold, double score_threshold) {
	int64_t* selected_indices_base = (int64_t*)selected_indices->base;

	double* boxes_base = (double*)boxes->base;
	uint32_t boxes_units[3];
	boxes_units[2] = 1;
	boxes_units[1] = boxes->lengths[2] * boxes_units[2];
	boxes_units[0] = boxes->lengths[1] * boxes_units[1];

	double* scores_base = (double*)scores->base;
	uint32_t scores_units[3];
	scores_units[2] = 1;
	scores_units[1] = scores->lengths[2] * scores_units[2];
	scores_units[0] = scores->lengths[1] * scores_units[1];

	uint32_t batch_count = scores->lengths[0];
	uint32_t class_count = scores->lengths[1];
	uint32_t coord_count = scores->lengths[2];
	int32_t index[batch_count][class_count][coord_count];

	// Make index
	for(uint32_t batch = 0; batch < batch_count; batch++) {
		for(uint32_t klass = 0; klass < class_count; klass++) {
			for(uint32_t i = 0; i < coord_count; i++) {
				double score = -FLT_MAX;
				int32_t idx = -1;

				for(uint32_t coord = 0; coord < coord_count; coord++) {
					double s = SCORES(batch, klass, coord);
					
					if(s > score || idx < 0) {
						for(uint32_t j = 0; j < i; j++) {
							if(index[batch][klass][j] == (int32_t)coord) {
								goto found;
							}
						}
						score = s;
						idx = coord;
found:
						;
					}
				}

				index[batch][klass][i] = idx;
			}
		}
	}

	// non max suppression
	for(uint32_t batch = 0; batch < batch_count; batch++) {
		for(uint32_t klass = 0; klass < class_count; klass++) {
			for(uint32_t coord = 0; coord < coord_count; coord++) {
				int32_t bbox_max = index[batch][klass][coord];
				if(bbox_max < 0)
					continue;

				for(uint32_t i = coord + 1; i < coord_count; i++) {
					int32_t bbox_cur = index[batch][klass][i];
					if(bbox_cur < 0)
						continue;

					double score_cur = SCORES(batch, klass, bbox_cur);
					if(score_cur < score_threshold)
						continue;

					double iou = iou_float64(BOXES(batch, bbox_max), BOXES(batch, bbox_cur));
					if(iou < iou_threshold)
						continue;

					index[batch][klass][i] = -1;
				}
			}
		}
	}

	// output
	uint32_t selected_idx = 0;
	for(uint32_t batch = 0; batch < batch_count; batch++) {
		for(uint32_t klass = 0; klass < class_count; klass++) {
			uint32_t klass_count = 0;

			for(uint32_t coord = 0; coord < coord_count; coord++) {
				int32_t idx = index[batch][klass][coord];
				if(idx < 0)
					continue;

				if(klass_count++ >= max_output_boxes_per_class || selected_idx++ >= selected_indices->lengths[0])
					break;

				*selected_indices_base++ = batch;
				*selected_indices_base++ = klass;
				*selected_indices_base++ = idx;
			}
		}
	}
}

static bool NonMaxSuppression_exec(uintptr_t* stack) {
	connx_Tensor* selected_indices = (void*)stack[1];	// batch_index, 3(num_batches, class_index, box_index)
	connx_Tensor* boxes = (void*)stack[2];				// num_batches, spactial_dimension, 4(coords)
	connx_Tensor* scores = (void*)stack[3];				// num_batches, num_classes, spatial_dimension
	connx_Tensor* max_output_boxes_per_class = (void*)stack[4];	// scalar
	connx_Tensor* iou_threshold = (void*)stack[5];				// scalar
	connx_Tensor* score_threshold = (void*)stack[6];			// scalar

	switch(boxes->elemType) {
		case connx_DataType_FLOAT32:
			process_float32(selected_indices, boxes, scores, 
					*(int64_t*)max_output_boxes_per_class->base, 
					*(float*)iou_threshold->base, 
					*(float*)score_threshold->base);
			break;
		case connx_DataType_FLOAT64:
			process_float64(selected_indices, boxes, scores, 
					*(int64_t*)max_output_boxes_per_class->base, 
					*(double*)iou_threshold->base, 
					*(double*)score_threshold->base);
			break;
		default:
			abort();
	}

	return true;
}

bool connx_opset_NonMaxSuppression_init() {
	connx_Operator_add("NonMaxSuppression", 1, 5, 1, NonMaxSuppression_resolve, NonMaxSuppression_exec,
		connx_DataType_TENSOR_INT64,	// selected_indices
		connx_DataType_TENSOR_FLOAT,	// boxes
		connx_DataType_TENSOR_FLOAT,	// scores
		connx_DataType_TENSOR_INT64,	// max_output_boxes_per_class
		connx_DataType_TENSOR_FLOAT,	// iou_threshold
		connx_DataType_TENSOR_FLOAT,	// score_threshold
		"center_point_box", connx_DataType_INT64, 0);		// center_point_box

	return true;
}
