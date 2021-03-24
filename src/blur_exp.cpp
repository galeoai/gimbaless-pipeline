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
    cv::Mat offset;
    void operator()(cv::Mat image) {
        auto h_image_in = Zerocopy::gpu<uint8_t>(image);
        auto output = Zerocopy::gpu<uint8_t>(IIR);
	auto h_offset = Zerocopy::gpu<uint8_t>(offset);
        process(h_image_in, output, h_offset, output);
        output.device_sync();
	IIR.copyTo(image);
	Zerocopy::release<uint8_t>(image);
	Zerocopy::release<uint8_t>(IIR);
	Zerocopy::release<uint8_t>(offset);
        return;
    }
} iir;
///////////////////////////////////////////////////////////////////////////////
//                                   simple                                  //
///////////////////////////////////////////////////////////////////////////////
struct simple {
    cv::Mat IIR;
    cv::Mat offset;
    void operator()(cv::Mat image) {
        image = image - offset;
        cv::addWeighted(IIR, 0.8, image, 0.2, 0.0, IIR);
        IIR.copyTo(image);
        return;
    }
} s;
///////////////////////////////////////////////////////////////////////////////
//                                   bypass                                  //
///////////////////////////////////////////////////////////////////////////////
void bypass(cv::Mat image) {
    return;
};
///////////////////////////////////////////////////////////////////////////////
//                                 alg callbacks                             //
///////////////////////////////////////////////////////////////////////////////
void buttoncallbackReg(int state, void *userdata) {
    auto config = (kaya_config *)userdata;
    if (state == 1) {
        iir.offset = config->offset.clone();
        iir.IIR = cv::Mat::zeros(config->width, config->height, CV_8UC1);
	usleep(10000);
        config->process = iir;
    }
}

void buttoncallbackIIR(int state, void *userdata) {
    auto config = (kaya_config *)userdata;
    if (state == 1) {
        s.offset = config->offset.clone();
        s.IIR = cv::Mat::zeros(config->width, config->height, CV_8UC1);
	usleep(10000);
        config->process = s;
    }
}

void buttoncallbackNO(int state, void *userdata) {
    usleep(2000);
    auto config = (kaya_config *)userdata;
    if (state == 1) {
        config->process = bypass;
    }
}

///////////////////////////////////////////////////////////////////////////////
//                                NUC callback                               //
///////////////////////////////////////////////////////////////////////////////
void buttoncallbackNUC(int state, void *userdata) {
    auto config = (kaya_config *)userdata;
    config->process = bypass;
    cv::Mat tmp = cv::Mat::zeros(config->width, config->height, CV_16UC1);
    for (auto i = 0; i < 50; ++i) {
        usleep(2000);
        tmp += config->image;
    }
    tmp /= 50;
    tmp.convertTo(config->offset, CV_8UC1);
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
    config.fps = 200;
    config.image = cv::Mat::zeros(config.width, config.height, CV_8UC1);
    config.offset = cv::Mat::zeros(config.width, config.height, CV_8UC1);
    config.process = bypass;

    setup(config);

    start(config);

    auto dis = config.image.clone();
    auto clahe = cv::createCLAHE(5.0, cv::Size(128, 128));

    namedWindow("image", cv::WINDOW_AUTOSIZE);
    // one point nuc
    cv::createButton("One point NUC", buttoncallbackNUC, (void *)&config, cv::QT_PUSH_BUTTON, 0);
    // alg button
    cv::createButton("No-alg", buttoncallbackNO, (void *)&config, cv::QT_RADIOBOX, 1);
    cv::createButton("IIR-only", buttoncallbackIIR, (void *)&config, cv::QT_RADIOBOX, 0);
    cv::createButton("Reg-GPU", buttoncallbackReg, (void *)&config, cv::QT_RADIOBOX, 0);

    while (true) {
	try {
	    cv::equalizeHist(config.image, dis);
	    //clahe->apply(config.image, dis);
	    cv::imshow("image", dis);

	    char c = (char)cv::waitKey(30);
	    if (c == 27) break;
	} catch(cv::Exception& e){
	    stop(config);
	    const char* err_msg = e.what();
	    std::cout << "exception caught: " << err_msg << std::endl;
	}
    }

    stop(config);
    std::cout << "Done!\n";

    return 0;
}
