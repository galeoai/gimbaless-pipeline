#include "halide_benchmark.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"

#include "process.h"

int main(int argc, char *argv[]) {
    //load images
    cv::Mat image1, image2, image_out;
    image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);  // 8bit
    image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    image_out = cv::Mat::zeros(image1.rows, image1.cols, image1.type());
    if (!image1.data && !image2.data) {
        printf("No images data \n");
        return -1;
    }
    auto im1_ptr = image1.ptr<uint8_t>(0);
    cudaMallocManaged(&im1_ptr,
		      image1.total() * image1.elemSize(),
		      cudaMemAttachHost);
    Halide::Runtime::Buffer<uint8_t> h_image1(im1_ptr,image1.rows,image1.cols);

    auto im2_ptr = image2.ptr<uint8_t>(0);
    cudaMallocManaged(&im2_ptr,
		      image1.total() * image1.elemSize(),
		      cudaMemAttachHost);
    Halide::Runtime::Buffer<uint8_t> h_image2(im2_ptr,image2.rows,image2.cols);

    auto out_ptr = image2.ptr<uint8_t>(0);
    cudaMallocManaged(&out_ptr,
		      image1.total() * image1.elemSize(),
		      cudaMemAttachHost);
    Halide::Runtime::Buffer<uint8_t> output(out_ptr,image_out.rows,image_out.cols);
    cv::Mat offset = cv::Mat::zeros(image1.rows, image1.cols, image1.type());
    auto off_ptr = offset.ptr<uint8_t>(0);
    cudaMallocManaged(&off_ptr,
		      image1.total() * image1.elemSize(),
		      cudaMemAttachHost);
    Halide::Runtime::Buffer<uint8_t> h_offset(off_ptr,offset.rows,offset.cols);


    // benchmark
    double auto_schedule_off = Halide::Tools::benchmark(5, 10, [&]() {
        process(h_image1, h_image2,h_offset, output);
        output.device_sync();
    });
    printf("Manual schedule: %gms\n", auto_schedule_off * 1e3);

    if (auto_schedule_off * 1e3 > 5) {
        std::cout << "pipline is slower that 5ms"
                  << "\n";
        return 1;
    }
    std::cout << "Success!"
              << "\n";
    return 0;
}
