#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>


using cv::Mat;

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

template<typename T> inline
Halide::Runtime::Buffer<T> mem::to_halide(Mat image) {
    auto image_ptr = image.ptr<T>(0); // will only work for continuous images
    Halide::Runtime::Buffer<T> h_image(nullptr, image.rows, image.cols);
    h_image.device_wrap_native(halide_cuda_device_interface(),
                               (uintptr_t)image_ptr);
    h_image.set_host_dirty();
    return h_image;
}

template<typename T> inline
Mat mem::gpu(Mat image) {
    auto image_ptr = image.ptr<T>(0); // will only work for continuous images
    void *image_gpu;
    size_t size = image.total() * image.elemSize();
    gpuErrchk(cudaHostAlloc(&image_gpu, size,
			    cudaHostAllocDefault));
    cudaMemcpy(image_gpu, (void *)image_ptr, size, cudaMemcpyHostToDevice);
    return Mat(image.rows, image.cols, CV_8UC1, image_gpu);
}


