#include "../src/kaya_interface.h"

#include <functional>
#include <opencv2/opencv.hpp>

void bypass(cv::Mat image){
    return;
}

int main(int argc, char *argv[]) {
   kaya_config config;
   config.width = 1024;
   config.height = 1024;
   config.offsetx = 512;
   config.offsety = 0;
   config.pixelFormat = "Mono8";
   config.grabberIndex = 0;
   config.cameraIndex = 0;
   config.exposure = 1000.0;
   config.fps = 20;
   config.image = cv::Mat::zeros(config.width, config.height, CV_8UC1);
   config.offset = cv::Mat::zeros(config.width, config.height, CV_8UC1);

   
   setup(config);
   
   start(config);
   
   while(true){
       cv::imshow("image", config.image);
       char c = (char)cv::waitKey(10);
       if (c == 27) break;
   }
   
   stop(config);
   std::cout << "Done!" << "\n";

	
   return 0;
}
