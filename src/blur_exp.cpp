#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>
#include <unistd.h>

// kaya interface
#include "kaya_interface.h"

// halide pipeline
#include "process.h"
#include "zerocopy.h"

///////////////////////////////////////////////////////////////////////////////
//                                  IIR alg                                  //
///////////////////////////////////////////////////////////////////////////////
struct IIR_alg {
    cv::Mat IIR;
    void operator()(cv::Mat image) {
        auto h_image_in = Zerocopy::gpu<uint8_t>(image);
        auto output = Zerocopy::gpu<uint8_t>(IIR);
        process(h_image_in, output, output);
        output.device_sync();
	IIR.copyTo(image);
        return;
    }
};
///////////////////////////////////////////////////////////////////////////////
//                                   simple                                  //
///////////////////////////////////////////////////////////////////////////////
struct simple {
    cv::Mat IIR;
    void operator()(cv::Mat image) {
        cv::addWeighted(IIR, 0.7, image, 0.3, 0.0, IIR);
	IIR.copyTo(image);
        return;
    }
};
///////////////////////////////////////////////////////////////////////////////
//                                   bypass                                  //
///////////////////////////////////////////////////////////////////////////////
void bypass(cv::Mat image) {
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
    config.fps = 300;
    config.image = cv::Mat::zeros(config.width, config.height, CV_8UC1);

    //IIR_alg iir;
    //iir.IIR = cv::Mat::zeros(config.width, config.height, CV_8UC1);
    //config.process = iir;
    //config.process = bypass;
    simple s;
    s.IIR = cv::Mat::zeros(config.width, config.height, CV_8UC1);
    config.process = s;

    setup(config);
    std::cout << "exposure: " << config.exposure << "\n";

    start(config);

    auto dis = config.image.clone();
    while (true) {
	cv::equalizeHist(config.image, dis);
        cv::imshow("image", dis);
        char c = (char)cv::waitKey(10);
        if (c == 27) break;
    }

    stop(config);
    std::cout << "Done!\n";

    return 0;
}
