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
    // load images 
    cv::Mat image1,image2;
    image1 = cv::imread(argv[1], cv::IMREAD_ANYDEPTH);
    image2 = cv::imread(argv[2], cv::IMREAD_ANYDEPTH);
    
    if ( !image1.data && !image2.data )
    {
        printf("No images data \n");
        return -1;
    }
    // image1 => halide buffer with zerocopy
    auto im1 = image1.ptr<uint8_t>(0);
    int *im1_gpu;
    cudaHostRegister(im1, image1.total() * image1.elemSize() , cudaHostRegisterMapped);
    gpuErrchk(cudaHostGetDevicePointer((void **)&im1_gpu, (void *)im1, 0));
    Halide::Runtime::Buffer<uint8_t> h_image1(nullptr, image1.rows,image1.cols);
    h_image1.device_wrap_native(halide_cuda_device_interface(),(uintptr_t)im1_gpu);
    h_image1.set_host_dirty();
    
    // image2 => halide buffer with zerocopy
    auto im2 = image1.ptr<uint8_t>(0);
    int *im2_gpu;
    cudaHostRegister(im2, image2.total() * image2.elemSize() , cudaHostRegisterMapped);
    gpuErrchk(cudaHostGetDevicePointer((void **)&im2_gpu, (void *)im2, 0));
    Halide::Runtime::Buffer<uint8_t> h_image2(nullptr, image2.rows,image2.cols);
    h_image2.device_wrap_native(halide_cuda_device_interface(),(uintptr_t)im2_gpu);
    h_image2.set_host_dirty();

    //output buffer
    cv::Mat image_out = cv::Mat::zeros(image1.rows,image1.cols,image1.type()); 
    auto im_out = image_out.ptr<uint8_t>(0);
    int *im_out_gpu;
    cudaHostRegister(im_out, image_out.total() * image_out.elemSize() , cudaHostRegisterMapped);
    gpuErrchk(cudaHostGetDevicePointer((void **)&im_out_gpu, (void *)im_out, 0));
    
    Halide::Runtime::Buffer<uint8_t> output(nullptr, image_out.rows,image_out.cols);
    output.device_wrap_native(halide_cuda_device_interface(),(uintptr_t)im_out_gpu);

    // benchmark
    double auto_schedule_off = Halide::Tools::benchmark(2, 5, [&]() {
	gpu_only(h_image1,h_image2, output);
	output.device_sync();
    });
    printf("Manual schedule: %gms\n", auto_schedule_off * 1e3);

    cv::imwrite("out.tiff", image_out);
    std::cout << "Done!" << "\n";

    return 0;
}
