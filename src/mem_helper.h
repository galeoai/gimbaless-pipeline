#ifndef MEM_HELPER_H
#define MEM_HELPER_H

#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }


inline void gpuAssert(cudaError_t code, const char *file, int line, 
    bool abort = false);

template<typename T>
int Ff();

namespace mem {

    template<typename T>
    inline Halide::Runtime::Buffer<T> to_halide(cv::Mat image);

    template<typename T>
    inline cv::Mat gpu(cv::Mat image);

}  // namespace mem

#include "mem_helper.tpp"
#endif
