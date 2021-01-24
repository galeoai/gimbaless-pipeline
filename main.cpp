#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include "opencv2/opencv.hpp"

#include <cstdlib>

// halide pipeline
#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"
#include "halide_benchmark.h"
#include "process.h"

#include "zerocopy.h"

int main(int argc, char *argv[])
{
    // load images 
    cv::Mat image1,image2;
    image1 = cv::imread(argv[1], cv::IMREAD_ANYDEPTH);
    image2 = cv::imread(argv[2], cv::IMREAD_ANYDEPTH);
    cv::Mat image_out = cv::Mat::zeros(image1.rows,image1.cols,image1.type()); 
    //std::vector<cv::String> filenames;
    //cv::glob(strcat(argv[3],"/*.tif"), filenames);
    //std::vector<cv::Mat> images;
    //for (auto &file : filenames) {
    //	std::cout << file << "\n";
    //	images.push_back(cv::imread(file, cv::IMREAD_ANYDEPTH));
    //}
    if ( !image1.data && !image2.data )
    {
        printf("No images data \n");
        return -1;
    }
    
    auto h_image1 = Zerocopy::gpu<uint8_t>(image1);
    auto h_image2 = Zerocopy::gpu<uint8_t>(image2);
    auto output	  = Zerocopy::gpu<uint8_t>(image_out);

    // benchmark
    double auto_schedule_off = Halide::Tools::benchmark(2, 5, [&]() {
	process(h_image1,h_image2, output);
	output.device_sync();
    });
    printf("Manual schedule: %gms\n", auto_schedule_off * 1e3);

    cv::imwrite("out.tiff", image_out);
    std::cout << "Done!" << "\n";

    return 0;
}
