#include <bits/stdint-uintn.h>
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
#include "mem_helper.h"

///////////////////////////////////////////////////////////////////////////////
//                                  IIR alg                                  //
///////////////////////////////////////////////////////////////////////////////
struct IIR_alg {
    cv::Mat IIR;
    cv::Mat offset;
    cv::Mat gain;
    void operator()(cv::Mat image) {
	auto output = mem::to_halide<uint8_t>(IIR);
	auto h_offset = mem::to_halide<uint8_t>(offset);
	auto h_image = mem::to_halide<uint8_t>(image);
	auto h_gain = mem::to_halide<float_t>(gain);
        process(h_image, output, h_offset, h_gain, output);
	h_image.device_sync();
	IIR.copyTo(image);
        return;
    }
} iir;
///////////////////////////////////////////////////////////////////////////////
//                                   simple                                  //
///////////////////////////////////////////////////////////////////////////////
struct simple {
    cv::Mat IIR;
    cv::Mat offset;
    cv::Mat gain;
    void operator()(cv::Mat image) {
        image-=offset;
	cv::multiply(image,gain,image,1.0,CV_8UC1);
	//image*=gain; 
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
	iir.offset = mem::gpu<uint8_t>(config->offset);
	iir.gain = mem::gpu<float_t>(config->gain);
	iir.IIR = mem::gpu<uint8_t>(config->image);
	usleep(10000);
        config->process = iir;
    }
}

void buttoncallbackIIR(int state, void *userdata) {
    auto config = (kaya_config *)userdata;
    if (state == 1) {
        s.offset = config->offset.clone();
	s.gain = config->gain.clone();
        s.IIR = config->image.clone();
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
void buttoncallbackDark(int state, void *userdata) {
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
void buttoncallbackFlat(int state, void *userdata) {
    auto config = (kaya_config *)userdata;
    config->process = bypass;
    cv::Mat tmp = cv::Mat::zeros(config->width, config->height, CV_16UC1);
    for (auto i = 0; i < 50; ++i) {
        usleep(2000);
        tmp += config->image;
    }
    tmp /= 50;
    //tmp.convertTo(config->offset, CV_8UC1);
    auto nominator = cv::mean(tmp)-cv::mean(config->offset);
    auto diff = config->gain.clone();
    cv::subtract(tmp,config->offset,diff,cv::noArray(), CV_32F);
    config->gain = nominator[0]/diff;
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
    config.offset = cv::Mat::zeros(config.width, config.height, CV_8UC1);
    config.gain = cv::Mat::zeros(config.width, config.height, CV_32F);
    config.process = bypass;

    setup(config);

    start(config);

    auto dis = config.image.clone();
    auto clahe = cv::createCLAHE(5.0, cv::Size(128, 128));

    namedWindow("image", cv::WINDOW_AUTOSIZE);
    // one point nuc
    cv::createButton("NUC[Dark]", buttoncallbackDark, (void *)&config, cv::QT_PUSH_BUTTON, 0);
    cv::createButton("NUC[Flat]", buttoncallbackFlat, (void *)&config, cv::QT_PUSH_BUTTON, 0);
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
