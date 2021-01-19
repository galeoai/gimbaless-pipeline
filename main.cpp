#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include "opencv2/opencv.hpp"

#include <cstdlib>

// cuda zerocopy
#include <cuda_runtime.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// halide pipeline
#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"
#include "halide_benchmark.h"
#include "gpu_only.h"

int main(int argc, char *argv[])
{
    printf("Hello!\n");
    cv::Mat image,image1;
    image = cv::imread( argv[1], cv::IMREAD_ANYDEPTH);
    image1 = cv::imread( argv[2], cv::IMREAD_ANYDEPTH);

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    auto im = image.ptr<uint16_t>(0);

    int *im_gpu;
    cudaHostRegister(im, image.total() * image.elemSize() , cudaHostRegisterMapped);
    gpuErrchk(cudaHostGetDevicePointer((void **)&im_gpu, (void *)im, 0));
    Halide::Runtime::Buffer<uint16_t> input(nullptr, image.rows,image.cols);
    input.device_wrap_native(halide_cuda_device_interface(),(uintptr_t)im_gpu);
    input.set_host_dirty();

    auto im1 = image1.ptr<uint16_t>(0);

    int *im1_gpu;
    cudaHostRegister(im1, image1.total() * image1.elemSize() , cudaHostRegisterMapped);
    gpuErrchk(cudaHostGetDevicePointer((void **)&im1_gpu, (void *)im1, 0));
    Halide::Runtime::Buffer<uint16_t> input1(nullptr, image1.rows,image1.cols);
    input1.device_wrap_native(halide_cuda_device_interface(),(uintptr_t)im1_gpu);
    input1.set_host_dirty();

    
    cv::Mat image_out = cv::Mat::zeros(image.rows,image.cols,image.type()); 
    //cv::Mat image_out = cv::Mat::zeros(image.rows/32,image.cols/32,CV_32SC1);
    auto im_out = image_out.ptr<uint16_t>(0);

    int *im_out_gpu;
    cudaHostRegister(im_out, image_out.total() * image_out.elemSize() , cudaHostRegisterMapped);
    gpuErrchk(cudaHostGetDevicePointer((void **)&im_out_gpu, (void *)im_out, 0));
    
    Halide::Runtime::Buffer<uint16_t> output(nullptr, image_out.rows,image_out.cols);
    output.device_wrap_native(halide_cuda_device_interface(),(uintptr_t)im_out_gpu);
    
    
    
    //for (auto i=0; i < 100; ++i) {
    // 	gpu_only(input,input, output);
    //}

    double auto_schedule_off = Halide::Tools::benchmark(2, 5, [&]() {
        
	gpu_only(input,input1, output);
	output.device_sync();
    });
    printf("Manual schedule: %gms\n", auto_schedule_off * 1e3);
    // Verify output.
    //for (int y = 0; y < image.cols; y++) {
    // 	for (int x = 0; x < image.rows; x++) {
    // 	    if (im[x+y*image.rows] * 3 !=  im_out[x+y*image.rows]) {
    // 		printf("Error at %d, %d: 3*%d != %d\n", x, y, im[x+y*image.rows], im_out[x+y*image.rows]);
    // 		return -1;
    // 	    }
    // 	}
    //}
    //for (auto i=0; i<image_out.cols; ++i) { 
    // 	for (auto j=0; j<image_out.rows; ++j) { 
    // 	    std::cout << im_out[i+j*image_out.rows] << ", ";
    // 	}
    // 	std::cout << "\n";
    //}
    //std::cout << "cols: " << image_out.cols << "\n";
    //std::cout << "rows: " << image_out.rows << "\n";

    cv::imwrite("out.tiff", image_out);
    std::cout << "Success!" << "\n";

    return 0;
}
