#include "../src/kaya_interface.h"

#include <cstdlib>  //for uint8_t
#include <functional>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    kaya_config config;
    config.width = 1024;
    config.height = 1024;
    config.grabberIndex = 1;
    config.cameraIndex = 0;
    config.fps = 25;
    config.image = 255 * cv::Mat::ones(1024, 1024, CV_8UC1);

    setup(config);
    start(config);
    while (true) {
	if(config.totalFrames>100) {break;}
    };
    stop(config);
    std::cout << "\nSuccess" << "\n";

    return 0;
}
