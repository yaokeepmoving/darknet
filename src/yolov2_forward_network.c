#include "yolov2_forward_network.h"


// 4 layers in 1: convolution, batch-normalization, BIAS and activation
void forward_convolutional_layer_cpu(layer l, network_state state)
{
	int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;	// convolutional_out_height(l);
	int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;	// convolutional_out_width(l);
	int i;

	// fill zero (ALPHA)
	for (int i = 0; i < l.outputs; ++i) l.output[i] = 0;


	printf("\n l.size = %d \n", l.size);

	// l.n - number of filters on this layer
	// l.c - channels of input-array
	// l.h - height of input-array
	// l.w - width of input-array
	// l.size - width and height of filters (the same size for any filter)

	// 1. Convolution !!!

	// filter number ("omp parallel for" - automatic parallelization of loop by using OpenMP)
	int fil;
	#pragma omp parallel for
	for (fil = 0; fil < l.n; ++fil)
		// channel number
		for (int che = 0; che < l.c; ++che)
			// input - y
			for (int y = 0; y < l.h; ++y)
				// input - x
				for (int x = 0; x < l.w; ++x)
				{
					int const output_index = fil*l.w*l.h + y*l.w + x;
					int const weights_pre_index = fil*l.c*l.size*l.size + che*l.size*l.size;
					int const input_pre_index = che*l.w*l.h;
					float sum = 0;

					// filter - y
					for (int f_y = 0; f_y < l.size; ++f_y)
					{
						int input_y = y + f_y - l.pad;
						// filter - x
						for (int f_x = 0; f_x < l.size; ++f_x)
						{
							int input_x = x + f_x - l.pad;
							if (input_y < 0 || input_x < 0 || input_y >= l.h || input_x >= l.w) continue;
							int input_index = input_pre_index + input_y*l.w + input_x;
							int weights_index = weights_pre_index + f_y*l.size + f_x;

							sum += state.input[input_index] * l.weights[weights_index];
						}
					}

					l.output[output_index] += sum;
				}


	int const out_size = out_h*out_w;

	// 2. Batch normalization
	if (l.batch_normalize) {
		for (int f = 0; f < l.out_c; ++f) {
			for (int i = 0; i < out_size; ++i) {
				int index = f*out_size + i;
				l.output[index] = (l.output[index] - l.rolling_mean[f]) / (sqrt(l.rolling_variance[f]) + .000001f);
			}
		}

		// scale_bias
		for (int i = 0; i < l.out_c; ++i) {
			for (int j = 0; j < out_size; ++j) {
				l.output[i*out_size + j] *= l.scales[i];
			}
		}
	}


	// 3. Add BIAS
	for (int i = 0; i < l.n; ++i) {
		for (int j = 0; j < out_size; ++j) {
			l.output[i*out_size + j] += l.biases[i];
		}
	}

	// 4. Activation function (LEAKY or LINEAR)
	if (l.activation == LEAKY) {
		for (int i = 0; i < l.n*out_size; ++i) {
			l.output[i] = leaky_activate(l.output[i]);
		}
	}


	/*
	int out_h = convolutional_out_height(l);
	int out_w = convolutional_out_width(l);
	int i;

	fill_cpu(l.outputs*l.batch, 0, l.output, 1);

	int m = l.n;
	int k = l.size*l.size*l.c;
	int n = out_h*out_w;

	float *a = l.weights;
	float *b = state.workspace;
	float *c = l.output;

	// convolution as GEMM (as part of BLAS)
	for(i = 0; i < l.batch; ++i){
		im2col_cpu(state.input, l.c, l.h, l.w,
		l.size, l.stride, l.pad, b);
		gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
		c += n*m;
		state.input += l.c*l.h*l.w;
	}

	if(l.batch_normalize){
		forward_batchnorm_layer(l, state);
	}
	add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);

	activate_array(l.output, m*n*l.batch, l.activation);
	*/
}


// MAX pooling layer
void forward_maxpool_layer_cpu(const layer l, network_state state)
{
	int b, i, j, k, m, n;
	int w_offset = -l.pad;
	int h_offset = -l.pad;

	int h = l.out_h;
	int w = l.out_w;
	int c = l.c;

	for (b = 0; b < l.batch; ++b) {
		for (k = 0; k < c; ++k) {
			for (i = 0; i < h; ++i) {
				for (j = 0; j < w; ++j) {
					int out_index = j + w*(i + h*(k + c*b));
					float max = -FLT_MAX;
					int max_i = -1;
					for (n = 0; n < l.size; ++n) {
						for (m = 0; m < l.size; ++m) {
							int cur_h = h_offset + i*l.stride + n;
							int cur_w = w_offset + j*l.stride + m;
							int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
							int valid = (cur_h >= 0 && cur_h < l.h &&
								cur_w >= 0 && cur_w < l.w);
							float val = (valid != 0) ? state.input[index] : -FLT_MAX;
							max_i = (val > max) ? index : max_i;
							max = (val > max) ? val : max;
						}
					}
					l.output[out_index] = max;
					l.indexes[out_index] = max_i;
				}
			}
		}
	}
}


// route layer
void forward_route_layer_cpu(const layer l, network_state state)
{
	int i, j;
	int offset = 0;
	for (i = 0; i < l.n; ++i) {
		int index = l.input_layers[i];
		float *input = state.net.layers[index].output;
		int input_size = l.input_sizes[i];
		for (j = 0; j < l.batch; ++j) {
			copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
		}
		offset += input_size;
	}
}


// reorg layer
void forward_reorg_layer_cpu(const layer l, network_state state)
{
	float *out = l.output;
	float *x = state.input;
	int w = l.w;
	int h = l.h;
	int c = l.c;
	int batch = l.batch;
	
	int b, i, j, k;
	int out_c = c;

	for (b = 0; b < batch; ++b) {
		for (k = 0; k < c; ++k) {
			for (j = 0; j < h; ++j) {
				for (i = 0; i < w; ++i) {
					int in_index = i + w*(j + h*(k + c*b));
					int c2 = k % out_c;
					int offset = k / out_c;
					int w2 = i;
					int h2 = j + offset;
					int out_index = w2 + w*(h2 + h*(c2 + out_c*b));
					out[in_index] = x[out_index];
				}
			}
		}
	}
}





// ---- region layer ----


static void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output)
{
	int b;
	for (b = 0; b < batch; ++b) {
		int i;
		int count = 0;
		for (i = 0; i < hierarchy->groups; ++i) {
			int group_size = hierarchy->group_size[i];
			softmax(input + b*inputs + count, group_size, temp, output + b*inputs + count);
			count += group_size;
		}
	}
}


static void softmax_cpu(float *input, int n, float temp, float *output)
{
	int i;
	float sum = 0;
	float largest = -FLT_MAX;
	for (i = 0; i < n; ++i) {
		if (input[i] > largest) largest = input[i];
	}
	for (i = 0; i < n; ++i) {
		float e = exp(input[i] / temp - largest / temp);
		sum += e;
		output[i] = e;
	}
	for (i = 0; i < n; ++i) {
		output[i] /= sum;
	}
}
// ---


// region layer
void forward_region_layer_cpu(const layer l, network_state state)
{
	int i, j, b, t, n;
	int size = l.coords + l.classes + 1;
	memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));
#ifndef GPU
	//flatten(l.output, l.w*l.h, size*l.n, l.batch, 1);
	{
		float *x = l.output;
		int layer_size = l.w*l.h;
		int layers = size*l.n;
		int batch = l.batch;

		float *swap = calloc(layer_size*layers*batch, sizeof(float));
		int i, c, b;
		for (b = 0; b < batch; ++b) {
			for (c = 0; c < layers; ++c) {
				for (i = 0; i < layer_size; ++i) {
					int i1 = b*layers*layer_size + c*layer_size + i;
					int i2 = b*layers*layer_size + i*layers + c;
					swap[i2] = x[i1];
				}
			}
		}
		memcpy(x, swap, layer_size*layers*batch * sizeof(float));
		free(swap);
	}

#endif
	for (b = 0; b < l.batch; ++b) {
		for (i = 0; i < l.h*l.w*l.n; ++i) {
			int index = size*i + b*l.outputs;
			float x = l.output[index + 4];
			l.output[index + 4] = 1. / (1. + exp(-x));	// logistic_activate_cpu(l.output[index + 4]);
		}
	}


#ifndef GPU
	
	if (l.softmax_tree) {	// Yolo 9000
		for (b = 0; b < l.batch; ++b) {
			for (i = 0; i < l.h*l.w*l.n; ++i) {
				int index = size*i + b*l.outputs;
				softmax_tree(l.output + index + 5, 1, 0, 1, l.softmax_tree, l.output + index + 5);
			}
		}
	}
	else if (l.softmax) {	// Yolo v2
		for (b = 0; b < l.batch; ++b) {
			for (i = 0; i < l.h*l.w*l.n; ++i) {
				int index = size*i + b*l.outputs;
				softmax_cpu(l.output + index + 5, l.classes, 1, l.output + index + 5);
			}
		}
	}
#endif
}






void yolov2_forward_network_cpu(network net, network_state state)
{
	state.workspace = net.workspace;
	int i;
	for (i = 0; i < net.n; ++i) {
		state.index = i;
		layer l = net.layers[i];

		if (l.type == CONVOLUTIONAL) {
			forward_convolutional_layer_cpu(l, state);
			printf("\n CONVOLUTIONAL \n");
		}
		else if (l.type == MAXPOOL) {
			forward_maxpool_layer_cpu(l, state);
			printf("\n MAXPOOL \n");
		}
		else if (l.type == ROUTE) {
			forward_route_layer_cpu(l, state);
			printf("\n ROUTE \n");
		}
		else if (l.type == REORG) {
			forward_reorg_layer_cpu(l, state);
			printf("\n REORG \n");
		}
		else if (l.type == REGION) {
			forward_region_layer_cpu(l, state);
			printf("\n REGION \n");
		}
		else {
			printf("\n layer: %d \n", l.type);
		}


		state.input = l.output;
	}
}

