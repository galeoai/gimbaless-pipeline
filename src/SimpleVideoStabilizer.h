#ifndef SIMPLEVIDEOSTABILIZER_H
#define SIMPLEVIDEOSTABILIZER_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

class Tracker {
    std::vector<cv::Point2f> trackedFeatures;
    cv::Mat prev;

public:
    Tracker();
    bool freshStart;
    cv::Mat_<float> rigidTransform;
    void processImage(cv::Mat &img);
};

#endif
