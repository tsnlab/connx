#include <inttypes.h>
#include <strings.h>
#include <float.h>
#include <connx/operator.h>
#include <connx/backend.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

/**
 * @param A y1, x1, y2, x2
 * @param B y1, x1, y2, x2
 * @return iou
 */
static float iou_float32(float* A, float* B, int32_t center_point_box) {
	float A_y1 = A[0];
	float A_x1 = A[1];
	float A_y2 = A[2];
	float A_x2 = A[3];

	float B_y1 = B[0];
	float B_x1 = B[1];
	float B_y2 = B[2];
	float B_x2 = B[3];

	if(center_point_box != 0) {
		float half_height = A_y2 / 2;
		float half_width = A_x2 / 2;

		A_x2 = A_x1 + half_width;
		A_y2 = A_y1 + half_height;
		A_x1 -= half_width;
		A_y1 -= half_height;

		if(A_x1 > A_x2) {
			float tmp = A_x1;
			A_x1 = A_x2;
			A_x2 = tmp;
		}

		if(A_y1 > A_y2) {
			float tmp = A_y1;
			A_y1 = A_y2;
			A_y2 = tmp;
		}

		half_height = B_y2 / 2;
		half_width = B_x2 / 2;

		B_x2 = A_x1 + half_width;
		B_y2 = A_y1 + half_height;
		B_x1 -= half_width;
		B_y1 -= half_height;

		if(B_x1 > A_x2) {
			float tmp = B_x1;
			B_x1 = A_x2;
			B_x2 = tmp;
		}

		if(B_y1 > A_y2) {
			float tmp = B_y1;
			B_y1 = A_y2;
			B_y2 = tmp;
		}
	}

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
static double iou_float64(double* A, double* B, int32_t center_point_box) {
	double A_y1 = A[0];
	double A_x1 = A[1];
	double A_y2 = A[2];
	double A_x2 = A[3];

	double B_y1 = B[0];
	double B_x1 = B[1];
	double B_y2 = B[2];
	double B_x2 = B[3];

	if(center_point_box != 0) {
		float half_height = A_y2 / 2;
		float half_width = A_x2 / 2;

		A_x2 = A_x1 + half_width;
		A_y2 = A_y1 + half_height;
		A_x1 -= half_width;
		A_y1 -= half_height;

		if(A_x1 > A_x2) {
			double tmp = A_x1;
			A_x1 = A_x2;
			A_x2 = tmp;
		}

		if(A_y1 > A_y2) {
			double tmp = A_y1;
			A_y1 = A_y2;
			A_y2 = tmp;
		}

		half_height = B_y2 / 2;
		half_width = B_x2 / 2;

		B_x2 = A_x1 + half_width;
		B_y2 = A_y1 + half_height;
		B_x1 -= half_width;
		B_y1 -= half_height;

		if(B_x1 > A_x2) {
			double tmp = B_x1;
			B_x1 = A_x2;
			B_x2 = tmp;
		}

		if(B_y1 > A_y2) {
			double tmp = B_y1;
			B_y1 = A_y2;
			B_y2 = tmp;
		}
	}

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
		uint32_t* max_output_boxes_per_class, float iou_threshold, float score_threshold, 
		int32_t center_point_box) {
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

					float iou = iou_float32(BOXES(batch, bbox_max), BOXES(batch, bbox_cur), center_point_box);
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
			uint32_t max_output_boxes = max_output_boxes_per_class[klass];

			for(uint32_t coord = 0; coord < coord_count; coord++) {
				int32_t idx = index[batch][klass][coord];
				if(idx < 0)
					continue;

				if(klass_count++ >= max_output_boxes || selected_idx++ >= selected_indices->lengths[0])
					break;

				*selected_indices_base++ = batch;
				*selected_indices_base++ = klass;
				*selected_indices_base++ = idx;
			}
		}
	}
}

static void process_float64(connx_Tensor* selected_indices, connx_Tensor* boxes, connx_Tensor* scores,
		uint32_t* max_output_boxes_per_class, double iou_threshold, double score_threshold,
		int32_t center_point_box) {
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

					double iou = iou_float64(BOXES(batch, bbox_max), BOXES(batch, bbox_cur), center_point_box);
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
			uint32_t max_output_boxes = max_output_boxes_per_class[klass];

			for(uint32_t coord = 0; coord < coord_count; coord++) {
				int32_t idx = index[batch][klass][coord];
				if(idx < 0)
					continue;

				if(klass_count++ >= max_output_boxes || selected_idx++ >= selected_indices->lengths[0])
					break;

				*selected_indices_base++ = batch;
				*selected_indices_base++ = klass;
				*selected_indices_base++ = idx;
			}
		}
	}
}

bool opset_NonMaxSuppression(connx_Backend* backend, uint32_t counts, uint32_t* params) {
	connx_Tensor* selected_indices = CONNX_GET_OUTPUT(0);	// batch_index, 3(num_batches, class_index, box_index)
	connx_Tensor* boxes = CONNX_GET_INPUT(0);				// num_batches, spactial_dimension, 4(coords)
	connx_Tensor* scores = CONNX_GET_INPUT(1);				// num_batches, num_classes, spatial_dimension
	connx_Tensor* max_output_boxes_per_class = CONNX_GET_INPUT(2);	// array
	connx_Tensor* iou_threshold = CONNX_GET_INPUT(3);				// scalar
	connx_Tensor* score_threshold = CONNX_GET_INPUT(4);				// scalar
	connx_AttributeInt* center_point_box = CONNX_GET_ATTRIBUTE(0);	// 0 or 1

	// normalize max_output_boxes_per_class
	uint32_t max_output_boxes_per_class_length = scores->lengths[1];
	uint32_t max_output_boxes_per_class_base[max_output_boxes_per_class_length];

	if(max_output_boxes_per_class == NULL) {
		bzero(max_output_boxes_per_class_base, sizeof(uint32_t) * max_output_boxes_per_class_length);
	}

	if(center_point_box->value != 0) { // convert x, y, w, h to y1, x1, y2, x2
		switch(boxes->type) {
			case connx_FLOAT32:
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
			case connx_FLOAT64:
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
				backend->pal->error(backend->pal, "Not supported type: %u\n", boxes->type);
				return false;
		}
	} else {	// convert flipped coordinates
		switch(boxes->type) {
			case connx_FLOAT32:
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
			case connx_FLOAT64:
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
				backend->pal->error(backend->pal, "Not supported type: %u\n", boxes->type);
				return false;
		}
	}

	switch(boxes->type) {
		case connx_FLOAT32:
			process_float32(selected_indices, boxes, scores, 
					max_output_boxes_per_class_base,
					iou_threshold != NULL ? *(float*)iou_threshold->base : 0.0,
					score_threshold != NULL ? *(float*)score_threshold->base : -FLT_MAX,
					center_point_box->value);
			break;
		case connx_FLOAT64:
			process_float64(selected_indices, boxes, scores, 
					max_output_boxes_per_class_base, 
					iou_threshold != NULL ? *(float*)iou_threshold->base : 0.0,
					score_threshold != NULL ? *(float*)score_threshold->base : -DBL_MAX,
					center_point_box->value);
			break;
		default:
			backend->pal->error(backend->pal, "Not supported type: %u\n", boxes->type);
			return false;
	}

	return true;
}
