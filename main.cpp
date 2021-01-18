#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/cudafeatures2d.hpp"

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

#include "gpu_only.h"

int main(int argc, char *argv[])
{
    printf("Hello!\n");
    cv::Mat image;
    image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE);
    image.convertTo(image, CV_32SC1);

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    auto im = image.ptr<int>(0);

    int *im_gpu;
    cudaHostRegister(im, image.total() * image.elemSize() , cudaHostRegisterMapped);
    gpuErrchk(cudaHostGetDevicePointer((void **)&im_gpu, (void *)im, 0));
    Halide::Runtime::Buffer<int> input(nullptr, image.rows,image.cols);
    input.device_wrap_native(halide_cuda_device_interface(),(uintptr_t)im_gpu);
    input.set_host_dirty();
    
    cv::Mat image_out = cv::Mat::zeros(image.rows,image.cols,image.type());
    auto im_out = image_out.ptr<int>(0);

    int *im_out_gpu;
    cudaHostRegister(im_out, image_out.total() * image_out.elemSize() , cudaHostRegisterMapped);
    gpuErrchk(cudaHostGetDevicePointer((void **)&im_out_gpu, (void *)im_out, 0));
    
    Halide::Runtime::Buffer<int> output(nullptr, image_out.rows,image_out.cols);
    output.device_wrap_native(halide_cuda_device_interface(),(uintptr_t)im_out_gpu);
    
    
    
    //for (auto i=0; i < 100; ++i) {
    // 	gpu_only(input,input, output);
    //}
    //gpu_only(input,input, output);
    //output.device_sync();    
    // Verify output.
    //for (int y = 0; y < image.cols; y++) {
    // 	for (int x = 0; x < image.rows; x++) {
    // 	    if (im[x+y*image.rows] * 3 !=  im_out[x+y*image.rows]) {
    // 		printf("Error at %d, %d: 3*%d != %d\n", x, y, im[x+y*image.rows], im_out[x+y*image.rows]);
    // 		return -1;
    // 	    }
    // 	}
    //}

    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    cv::Mat tmp;
    cv::cuda::GpuMat gpu_image;
    image.convertTo(tmp,CV_8UC1);
    gpu_image.upload(tmp);
    cv::Ptr<cv::cuda::FastFeatureDetector> gpuFastDetector = cv::cuda::FastFeatureDetector::create(100, true, 2);

    std::vector<cv::KeyPoint> gpuKeypoints;


    cv::cuda::GpuMat gFrame;
    gpuFastDetector->detect(gpu_image, gpuKeypoints);
    
    printf("Success!\n");

    return 0;
}
