#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
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
    //cv::Mat image_out = cv::Mat::zeros(image1.rows,image1.cols,image1.type());
    cv::Mat image_out = cv::Mat::zeros(1024,1024,image1.type()); 
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
    //double auto_schedule_off = Halide::Tools::benchmark(2, 5, [&]() {
    // 	process(h_image1,h_image2, output);
    // 	output.device_sync();
    //});
    //printf("Manual schedule: %gms\n", auto_schedule_off * 1e3);
    // 
    //cv::imwrite("out.tiff", image_out);

    cv::VideoWriter out;
    out.open("appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=10.4.20.12 port=5000 ",
	     0,
	     16,
	     cv::Size (1024, 1024),
	     false);
    
    out.write(image_out);
    while (true) {
	//cv::imshow("output",image_out);
	//out.write(image_out);
	cv::randu(image_out, 0,255);
	out.write(image1);
	char c=(char)cv::waitKey(25);
	if(c==27)
	    break;
    }
    //std::cout << "Done!" << "\n";

    return 0;
}
