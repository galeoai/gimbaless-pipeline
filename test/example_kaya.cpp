#include "../src/kaya_interface.h"

#include <functional>
#include <opencv2/opencv.hpp>

struct IIR_alg {
    cv::Mat IIR;
    int height;
    int width;
    cv::Mat operator()(void* input) {
	cv::Mat image(height, width, CV_8UC1, input);
	auto h_image_in = Zerocopy::gpu<uint8_t>(image);
	auto output = Zerocopy::gpu<uint8_t>(IIR);
	process(h_image_in, output, output);
	output.device_sync();
	return IIR;
    }
};


cv::Mat bypass(cv::Mat image){return image;}

int main(int argc, char *argv[]) {
    kaya_config config;
    config.width = 1024;
    config.height = 1024;
    config.grabberIndex = 0;
    config.cameraIndex = 0;
    config.fps = 250;
    config.image = cv::Mat::zeros(1024, 1024, CV_8UC1);
    config.process = bypass;
    
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
