#include <bits/stdint-uintn.h>
#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>
#include <cstdlib>
#include <unistd.h>

// halide pipeline
#include "process.h"
#include "mem_helper.h"


int main(int argc, char *argv[]) {
    std::vector<cv::String> filenames;
    if (argc == 1) {
	std::cout << "usage ./app <dir> " << "\n";
	return 0;
    }
    cv::glob(strcat(argv[1], "/*.tif"), filenames);
    std::vector<cv::Mat> images;
    for (auto &file : filenames) {
        std::cout << file << "\n";
        images.push_back(cv::imread(file, cv::IMREAD_GRAYSCALE));
    }
    cv::Mat image_out = cv::Mat::zeros(images[0].rows, images[0].cols, images[0].type());

    cv::VideoWriter out;
    out.open("appsrc ! videoconvert ! x264enc bitrate=5000 ! rtph264pay ! udpsink host=10.4.20.16 port=5000 ",
             0,
             30,
             image_out.size(),
             false);
    auto gpu = [](cv::Mat img) {return mem::to_halide<uint8_t>(mem::gpu<uint8_t>(img));};
    
    int i = 0;
    auto tmp_image = cv::Mat(images[0].rows, images[0].cols, images[0].type());
    auto noise = cv::Mat(images[0].rows, images[0].cols, images[0].type());
    auto h_image_in = gpu(tmp_image);
    cv::Mat offset = cv::Mat::zeros(images[0].rows, images[0].cols, images[0].type());
    auto h_offset = gpu(offset);
    auto output = gpu(image_out);
    //out.write(images[0]);
    while (true) {
        cv::randu(noise, -30, 30);
        //std::cout << i << "\n";
        tmp_image = images[i] + noise;
        //i=1;
        i++;

        //h_image_in = Zerocopy::gpu<uint8_t>(images[i]);
        //h_image_in = gpu(tmp_image);
        //process(h_image_in, output, h_offset, output);
        //output.device_sync();
        //out.write(image_out);


        //out << tmp_image;
        //cv::imshow("gimbaless", tmp_image);
        //cv::waitKey(0);
        //out << images[i];
        usleep(1000 * 20);
        i %= images.size();
    }
    std::cout << "Done!" << "\n";
    //cv::imwrite("out.tiff", image_out);

    return 0;
}
