#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>
#include "opencv2/opencv.hpp"

#include <cstdlib>
#include <unistd.h>

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
    //cv::Mat image1,image2;
    //image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE); // 8bit
    //image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    //if ( !image1.data && !image2.data )
    //{
    //    printf("No images data \n");
    //    return -1;
    //}
    //auto h_image1 = Zerocopy::gpu<uint8_t>(image1);
    //auto h_image2 = Zerocopy::gpu<uint8_t>(image2);
    //cv::Mat image_out = cv::Mat::zeros(image1.rows,image1.cols,image1.type());


    std::vector<cv::String> filenames;
    cv::glob(strcat(argv[1],"/*.tif"), filenames);
    std::vector<cv::Mat> images;
    for (auto &file : filenames) {
	std::cout << file << "\n";
	images.push_back(cv::imread(file, cv::IMREAD_GRAYSCALE));
    }
    cv::Mat image_out = cv::Mat::zeros(images[0].rows,images[0].cols,images[0].type());

    cv::VideoWriter out;
    //out.open("appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=5000 speed-preset=superfast ! rtph264pay ! udpsink host=10.4.20.12 port=5000 ",
    // 	     0,
    // 	     10,
    // 	     image_out.size(),
    // 	     false);
    out.open("appsrc ! videoconvert ! x264enc bitrate=5000 ! rtph264pay ! udpsink host=10.4.20.16 port=5000 ",
     	     0,
     	     25,
     	     image_out.size(),
     	     false);

    //out.write(image_out);

    // benchmark
    //double auto_schedule_off = Halide::Tools::benchmark(2, 5, [&]() {
    // 	process(h_image1,h_image2, output);
    // 	output.device_sync();
    //});
    //printf("Manual schedule: %gms\n", auto_schedule_off * 1e3);
    
    int i = 0;
    auto tmp_image = images[0];
    auto noise = images[0];
    auto h_image_in = Zerocopy::gpu<uint8_t>(tmp_image);
    auto output	= Zerocopy::gpu<uint8_t>(image_out);
    //out.write(images[0]);
    while (true) {
	cv::randu(noise, 0,30);
	std::cout << i << "\n";
	tmp_image = images[i];
	i++;

	//h_image_in = Zerocopy::gpu<uint8_t>(tmp_image);
	//process(h_image_in, h_image_in, output);
	out.write(images[i]);

	//out << tmp_image;
	//out << images[i];
	usleep(1000*2);
	i %= 5;
    }
    //std::cout << "Done!" << "\n";
    
    //cv::imwrite("out.tiff", image_out);

    return 0;
}
