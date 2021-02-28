#include "../src/kaya_interface.h"

#include <cstdlib>  //for uint8_t
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // for memcpy
#include <functional>


#include "process.h"
#include "../src/zerocopy.h"

void process1(alg_args args, void* input){
    cv::Mat image(args.height, args.width, CV_8UC1, input);
    auto h_image_in = Zerocopy::gpu<uint8_t>(image);
    auto output = Zerocopy::gpu<uint8_t>(args.IIR);
    process(h_image_in, output, output);
    output.device_sync();
}

int main(int argc, char *argv[]) {
    kaya_config config;
    config.width = 1024;
    config.height = 1024;
    config.grabberIndex = 0;
    config.cameraIndex = 0;
    config.fps = 300;
    config.image = 255*cv::Mat::ones(1024,1024,CV_8UC1);
    config.process = process1;

    alg_args args;
    args.IIR = cv::Mat::zeros(1024,1024,CV_8UC1);
    args.height = 1024;
    args.width = 1024;

    config.args = args;
    setup(config);
    start(config);
    while(true){
	cv::imshow( "image", config.image );
	char c=(char)cv::waitKey(10);
	if(c==27) break;
    };
    stop(config);

    printf("\nSuccess!\n");
    return 0;
}
