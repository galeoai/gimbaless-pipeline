#ifndef KAYA_INTERFACE_H
#define KAYA_INTERFACE_H
#include "KYFGLib.h"
#include <functional>
#include <opencv2/opencv.hpp>
#include <string>

struct kaya_config {
    long long width = 0;
    long long height = 0;
    long long offsetx = 0;
    long long offsety = 0;

    std::string pixelFormat = "Mono8";
    double fps = 25;
    double exposure = 10000.0;
    void *args;
    std::function<void(cv::Mat)> process = nullptr;
    cv::Mat image;
    long long totalFrames = 0;
    long long buffSize = 0;
    int grabberIndex = 0;
    int cameraIndex = 0;
    FGHANDLE handle;
    STREAM_HANDLE streamHandle = 0;
    CAMHANDLE camHandleArray[KY_MAX_CAMERAS];
};

bool setup(kaya_config &config);
bool start(kaya_config &config);
bool stop(kaya_config &config);

#endif
