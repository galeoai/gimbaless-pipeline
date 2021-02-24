#include "../src/kaya_interface.h"

#include <cstdlib>  //for uint8_t
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // for memcpy


int main(int argc, char *argv[]) {
    kaya_config config;
    config.width = 1024;
    config.height = 1024;
    config.grabberIndex = 0;
    config.cameraIndex = 0;
    config.image = 255*cv::Mat::ones(1024,1024,CV_8UC1);
    setup(config);
    start(config);
    while(true){
	cv::imshow( "image", config.image );
	char c=(char)cv::waitKey(25);
	if(c==27) break;
    };
    stop(config);

    printf("\nSuccess!\n");
    return 0;
}
