#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdlib.h>
#include "opencv2\opencv.hpp"
#include <stdio.h>
#include <ctime>

#define RADIUS 1
#define F_HEIGHT 3
#define F_WIDTH 3

__global__ void convolutionKernel(float *d_conmatrix, uchar *d_out, uchar *d_in, int inW, int inH)
{
	//Position of the pixel that this thread is going to process.    
	int blockid = blockIdx.x + blockIdx.y * gridDim.x;
	int pos = blockid * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	float sum = 0.0;
	float current = 0.0;

	for (int i = -RADIUS; i <= RADIUS; i++) { //Rows
		for (int j = -RADIUS; j <= RADIUS; j++) { //Columns
			//Checks if the thread is trying to process a pixel out of bounds.
			if (blockIdx.x == 0 && (threadIdx.x + j < 0)) {
				current = 0;
			} else if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x + j) > blockDim.x - 1) {
				current = 0;
			} else {
				if (blockIdx.y == 0 && (threadIdx.y + i) < 0) {
					current = 0;
				}
				else if ((blockIdx.y == gridDim.y - 1) && (threadIdx.y + i) > blockDim.y - 1) {
					current = 0;
				} else {
					//Saves the pixel value in the variable.
					current = d_in[pos + (i * inW) + j];
				}
			}
			sum += current * d_conmatrix[F_WIDTH * (i + RADIUS) + (j + RADIUS)];
		}
	}

	//Sets the pixel of the new image.
	d_out[pos] = floor(sum);
}

__global__ void convolutionSharedKernel(float *d_conmatrix, uchar *d_out, uchar *d_in, int inW, int inH)
{
	//Shared memory for the convolution matrix.
	__shared__ float shared_matrix[F_HEIGHT * F_WIDTH];

	//Position of the pixel that this thread is going to process.    
	int blockid = blockIdx.x + blockIdx.y * gridDim.x;
	int pos = blockid * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	//Each pixel of the block copies one pixel of the convolution matrix to shared memory.
	shared_matrix[(threadIdx.y * blockDim.x) + threadIdx.x] = d_conmatrix[(threadIdx.y * blockDim.x) + threadIdx.x];

	__syncthreads();

	float sum = 0.0;
	float current = 0.0;

	for (int i = -RADIUS; i <= RADIUS; i++) { //Rows
		for (int j = -RADIUS; j <= RADIUS; j++) { //Columns
												  //Checks if the thread is trying to process a pixel out of bounds.
			if (blockIdx.x == 0 && (threadIdx.x + j < 0)) {
				current = 0;
			}
			else if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x + j) > blockDim.x - 1) {
				current = 0;
			}
			else {
				if (blockIdx.y == 0 && (threadIdx.y + i) < 0) {
					current = 0;
				}
				else if ((blockIdx.y == gridDim.y - 1) && (threadIdx.y + i) > blockDim.y - 1) {
					current = 0;
				}
				else {
					//Saves the pixel value in the variable.
					current = d_in[pos + (i * inW) + j];
				}
			}
			sum += current * shared_matrix[F_WIDTH * (i + RADIUS) + (j + RADIUS)];
		}
	}

	//Sets the pixel of the new image.
	d_out[pos] = floor(sum);
}

void convolutionSerial(float *d_conmatrix, uchar *out, uchar *in, int inW, int inH) {
	

	for (int h = 0; h < inH; h++) { //Rows
		for (int l = 0; l < inW; l++) { //Columns

			float sum = 0.0;
			float current = 0.0;
			//Current pixel position.
			int pos = h * inW + l;

			for (int i = -RADIUS; i <= RADIUS; i++) { //Rows
				for (int j = -RADIUS; j <= RADIUS; j++) { //Columns
					if (l + j < 0) {
						current = 0;
					}
					else if (l + j > inW - 1) {
						current = 0;
					}
					else {
						if (h + i < 0) {
							current = 0;
						}
						else if (h + i > inH - 1) {
							current = 0;
						}
						else {
							//Saves the pixel value in the variable.
							current = in[pos + (i * inW) + j];
						}
					}
					sum += current * d_conmatrix[F_WIDTH * (i + RADIUS) + (j + RADIUS)];
				}
			}

			//Sets the pixel of the new image.
			out[pos] = floor(sum);
		}
	}
}

int main() {
	float *d_conmatrix;
	uchar *d_image, *d_out;
	uchar *h_out, *h_aux;
	clock_t t;

	//Gaussian box.
	float filter[F_HEIGHT * F_WIDTH] =
	{
		1.0/9.0, 1.0 / 9.0,1.0 / 9.0,
		1.0 / 9.0,1.0 / 9.0,1.0 / 9.0,
		1.0 / 9.0,1.0 / 9.0,1.0 / 9.0
	};

	//Load image data
	cv::Mat image = cv::imread("jupiter.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	int i_width = image.cols;
	int i_height = image.rows;
	
	if (!image.data) 
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	
	//Show original image.
	cv::namedWindow("Original window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original window", image);

	//Allocate host memory.
	h_out = (uchar*)malloc(i_height * i_width * sizeof(uchar));
	h_aux = (uchar*)malloc(i_height * i_width * sizeof(uchar));

	t = clock();
	//Applies three passes to maximize the blur effect. (Global memory filter)
	convolutionSerial(filter, h_out, image.data, i_width, i_height);
	convolutionSerial(filter, h_aux, h_out, i_width, i_height);
	convolutionSerial(filter, h_out, h_aux, i_width, i_height);
	t = clock() - t;

	//Print time
	std::cout << "Serial (3 convolutions):" << (float)t / CLOCKS_PER_SEC << " seconds\n" << std::endl;

	//Show processed image.
	cv::Mat res0(i_height, i_width, CV_8UC1, h_out);
	cv::namedWindow("Result window (serial)", cv::WINDOW_AUTOSIZE);
	cv::imshow("Result window (serial)", res0);

	//Alocate device memory.
	cudaMalloc((void**)&d_conmatrix, F_HEIGHT * F_WIDTH * sizeof(float));
	cudaMalloc((void**)&d_image, i_height * i_width * sizeof(uchar));
	cudaMalloc((void**)&d_out, i_height * i_width * sizeof(uchar));

	//Copy from host to device.
	cudaMemcpy(d_conmatrix, filter, F_HEIGHT * F_WIDTH * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_image, image.data, i_height * i_width * sizeof(uchar), cudaMemcpyHostToDevice);

	//Sizes of grid and block.
	dim3 block;
	dim3 grid;

	block.x = F_WIDTH;
	block.y = F_HEIGHT;
	grid.x = i_width / block.x;
	grid.y = i_height / block.y;

	t = clock();
	//Applies three passes to maximize the blur effect. (Global memory filter)
	convolutionKernel<<<grid, block>>> (d_conmatrix, d_out, d_image, i_width, i_height);
	convolutionKernel<<<grid, block >> > (d_conmatrix, d_out, d_out, i_width, i_height);
	convolutionKernel<<<grid, block >> > (d_conmatrix, d_out, d_out, i_width, i_height);

	cudaThreadSynchronize();
	t = clock() - t;

	std::cout << "Kernel Global Memory (3 convolutions):" << (float)t / CLOCKS_PER_SEC << " seconds\n" << std::endl;

	//Copy result from device to host.
	cudaMemcpy(h_out, d_out, i_height * i_width * sizeof(uchar), cudaMemcpyDeviceToHost);
	
	//Show processed image.
	cv::Mat res(i_height, i_width, CV_8UC1, h_out);
	cv::namedWindow("Result window (global)", cv::WINDOW_AUTOSIZE);
	cv::imshow("Result window (global)", res);

	t = clock();
	//Applies three passes to maximize the blur effect. (Global memory filter)
	convolutionKernel << <grid, block >> > (d_conmatrix, d_out, d_image, i_width, i_height);
	convolutionKernel << <grid, block >> > (d_conmatrix, d_out, d_out, i_width, i_height);
	convolutionKernel << <grid, block >> > (d_conmatrix, d_out, d_out, i_width, i_height);

	cudaThreadSynchronize();
	t = clock() - t;

	std::cout << "Kernel Shared Memory (3 convolutions):" << (float)t / CLOCKS_PER_SEC << " seconds\n" << std::endl;

	//Copy result from device to host.
	cudaMemcpy(h_out, d_out, i_height * i_width * sizeof(uchar), cudaMemcpyDeviceToHost);

	//Show processed image.
	cv::Mat res2(i_height, i_width, CV_8UC1, h_out);
	cv::namedWindow("Result window (shared)", cv::WINDOW_AUTOSIZE);
	cv::imshow("Result window (shared)", res2);

	cv::waitKey(0);
	return 0;
}