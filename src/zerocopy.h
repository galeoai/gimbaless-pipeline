#ifndef ZEROCOPY_H
#define ZEROCOPY_H

#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = false) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

namespace Zerocopy {
template<typename T>
inline Halide::Runtime::Buffer<T> gpu(cv::Mat image) {
    auto im1 = image.ptr<T>(0);
    int *im1_gpu;
    cudaHostRegister(im1, image.total() * image.elemSize(),
                     cudaHostRegisterMapped);
    gpuErrchk(cudaHostGetDevicePointer((void **)&im1_gpu, (void *)im1, 0));
    Halide::Runtime::Buffer<T> h_image(nullptr, image.rows, image.cols);
    h_image.device_wrap_native(halide_cuda_device_interface(),
                               (uintptr_t)im1_gpu);
    h_image.set_host_dirty();
    return h_image;
};

template<typename T>
inline Halide::Runtime::Buffer<T> to_halide(cv::Mat image) {
    auto image_ptr = image.ptr<T>(0);
    Halide::Runtime::Buffer<T> h_image(nullptr, image.rows, image.cols);
    h_image.device_wrap_native(halide_cuda_device_interface(),
                               (uintptr_t)image_ptr);
    h_image.set_host_dirty();
    return h_image;
};

template<typename T>
inline cv::Mat xavier(cv::Mat image) {
    auto image_ptr = image.ptr<T>(0);
    void *image_xavier;
    //cudaHostRegister(image_ptr, image.total() * image.elemSize(),
    //                 cudaHostRegisterMapped);
    //gpuErrchk(cudaHostGetDevicePointer((void **)&image_xavier,
    // 				       (void *)image_ptr, 0));
    gpuErrchk(cudaHostAlloc(&image_xavier,image.total() * image.elemSize(),
			    cudaHostAllocDefault));
    return cv::Mat(image.rows,image.cols,CV_8UC1,image_xavier);
};


}  // namespace Zerocopy

#endif
